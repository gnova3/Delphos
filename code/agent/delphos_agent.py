from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn import QNetwork
from agent.shared_representation import DeepSetEncoder, InteractionAwareEncoder, TermEncoder, ZStateConfig
from agent.replay_buffer import Transition
from mdp.state_manager import SpecificationTerm


class DelphosAgent(nn.Module):
    """Deep Q-Network agent for assisted choice model specification.

    Delphos combines:
    1. A shared state encoder that maps a set of specification terms into a fixed-dimensional latent representation.
    2. A policy Q-network used for action selection.
    3. A target Q-network used for stable Double-DQN updates.

    The agent takes a sequence of actions to propose a model specification, which is a set of SpecificationTerm objects.
    
    This set of SpecificationTerm objects is encoded into a latent space using either:
        - DeepSetEncoder (order invariant set encoding)
        - InteractionAwareEncoder (interaction-aware representation)

    The resulting latent state is used by the Q-network to estimate the Q-value of all available modelling actions, where actions may include:
        - Adding a new term to the specification
        - Modifying an existing term in the specification
        - Terminating the specification process

    Attributes:
        global_attribute_ids: Global attribute catalogue identifiers.
        global_transform_ids: Global transformation identifiers.
        global_taste_ids: Global taste identifiers.
        global_covariate_ids: Global covariate identifiers.
        num_actions: Size of the global action catalogue.
        policy_net (QNetwork): Online Q-network.
        target_net (QNetwork): Target Q-network used for Double-DQN targets.
        z_cfg: Shared representation configuration.
        encoder_kind: Encoder architecture (default: "deepset").
        head_flag: Whether the encoder should produce a dedicated state head.
        pooling: Pooling strategy used by the encoder.
        optimizer (torch.optim.Optimizer): Optimizer updating encoder and policy network parameters.
        learning_rate: Adam optimizer learning rate.
        discount_factor: Bellman discount factor γ.
        grad_clip_norm: Maximum gradient norm.
        target_soft_tau: Polyak update coefficient.
        device: Torch device.
        use_amp: Whether mixed precision training should be enabled.

        Raises:
            ValueError: If any configuration parameter is invalid.
    """

    def __init__(
        self,
        global_attribute_ids: tuple[int, ...],
        global_transform_ids: tuple[int, ...],
        global_taste_ids: tuple[int, ...],
        global_covariate_ids: tuple[int, ...],
        num_actions: int,
        z_cfg: Optional[ZStateConfig] = None,
        encoder_kind: str = "deepset",
        head_flag: bool = False,
        pooling: str = "mean",
        learning_rate: float = 1e-3,
        discount_factor: float = 0.90,
        grad_clip_norm: Optional[float] = 1.0,
        target_soft_tau: float = 5e-3,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
    ) -> None:        
        super().__init__()

        if len(global_attribute_ids) == 0:
            raise ValueError("global_attribute_ids must be non-empty.")
        if len(global_transform_ids) == 0:
            raise ValueError("global_transform_ids must be non-empty.")
        if len(global_taste_ids) == 0:
            raise ValueError("global_taste_ids must be non-empty.")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}.")
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError(f"discount_factor must be in [0,1], got {discount_factor}.")
        if target_soft_tau <= 0.0 or target_soft_tau > 1.0:
            raise ValueError(f"target_soft_tau must be in (0,1], got {target_soft_tau}.")

        # --- Inputs ---
        self.device          = torch.device("cpu") if device is None else torch.device(device)
        self.num_actions     = int(num_actions)
        self.discount_factor = float(discount_factor)
        self.learning_rate   = float(learning_rate)
        self.grad_clip_norm  = grad_clip_norm
        self.target_soft_tau = float(target_soft_tau)
        self.context_dim     = len(global_attribute_ids) + len(global_covariate_ids)

        # --- Z-state encoder ---
        if z_cfg is None:
            z_cfg = ZStateConfig(
                K=max(global_attribute_ids),
                T=max(global_transform_ids),
                G=max(global_taste_ids),
                C=max(global_covariate_ids) if len(global_covariate_ids) > 0 else 0,
                d_att=16, 
                d_tr=8,
                d_taste=8, 
                d_cov=16, 
                d_term=64,
                d_state=128,
                context_dim=self.context_dim,
                head_flag=head_flag,
                pooling=pooling,)
            
        self.z_cfg = z_cfg
        self.encoder_kind = encoder_kind.lower()
        self.term_encoder = TermEncoder(self.z_cfg)        
        if self.encoder_kind in {"deepset", "baseline"}:
            self.encoder = DeepSetEncoder(self.z_cfg, self.term_encoder)
        elif self.encoder_kind in {"interaction", "attention", "interaction_aware"}:
            self.encoder = InteractionAwareEncoder(self.z_cfg, self.term_encoder)
        else:
            raise ValueError("encoder_kind must be one of {'deepset', 'interaction'}.")
        self.encoder.to(self.device)
        
        # --- Q networks ---
        self.state_dim = self.z_cfg.d_state if self.z_cfg.head_flag else (self.z_cfg.d_term + self.z_cfg.context_dim)
        self.policy_net = QNetwork(input_size=self.state_dim, output_size=self.num_actions, hidden_layers=(256, 256),).to(self.device)
        self.target_net = QNetwork(input_size=self.state_dim, output_size=self.num_actions, hidden_layers=(256, 256),).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- Optimizer ---
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.policy_net.parameters()), lr=self.learning_rate,)
        self.autocast_device = ("cuda" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu")
        self.use_amp = bool(use_amp) and self.autocast_device == "cuda"
        self.scaler = torch.amp.GradScaler(self.autocast_device, enabled=self.use_amp)

    # ------------------------------------------------------------------
    #  utilities
    # ------------------------------------------------------------------  
    @staticmethod
    def _normalise_terms(terms: Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]],) -> list[SpecificationTerm]:
        """Convert a sequence of raw tuples into SpecificationTerm objects.

        Args:
            terms (Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]]): The terms to normalize.

        Returns:
            list[SpecificationTerm]: The normalized list of SpecificationTerm instances.
        """
        normalised: list[SpecificationTerm] = []
        for term in terms:
            if isinstance(term, SpecificationTerm):
                normalised.append(term)
            else:
                normalised.append(SpecificationTerm(attribute_id=int(term[0]), transformation_id=int(term[1]), taste_id=int(term[2]), covariate_id=int(term[3]),))
        return normalised
    
    def _terms_to_id_tensors(self, terms: Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]],) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Convert sequence of terms into distinct ID tensors.

        Args:
            terms (Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]]): The terms.

        Returns:
            tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]: 
                Tensors for attributes, transformations, tastes, and covariates.
        """
        normalised_terms = self._normalise_terms(terms)

        if len(normalised_terms) == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, empty, empty

        k_ids = torch.tensor([t.attribute_id for t in normalised_terms], dtype=torch.long, device=self.device)
        t_ids = torch.tensor([t.transformation_id for t in normalised_terms], dtype=torch.long, device=self.device)
        g_ids = torch.tensor([t.taste_id for t in normalised_terms], dtype=torch.long, device=self.device)
        i_ids = torch.tensor([t.covariate_id for t in normalised_terms], dtype=torch.long, device=self.device)
        return k_ids, t_ids, g_ids, i_ids

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def build_context_vector(self, runtime) -> torch.Tensor:
        """Construct the task context representation.

        The context vector encodes the modelling capabilities of the current task.
        This allows a single policy to operate across datasets with different
        variable structures.

        Args:
            runtime: TaskRuntime instance.

        Returns:
            torch.Tensor: Context vector of shape: [num_attributes + num_covariates]
        """
        attr_mask = runtime.task.attribute_mask.to(self.device).float()
        cov_mask = runtime.task.covariate_mask.to(self.device).float()
        return torch.cat([attr_mask, cov_mask], dim=0)      
    
    def encode_terms(self,terms: Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]],runtime,) -> torch.Tensor:        
        """Encode a model specification into a latent state representation.

        The encoder combines the specification terms with task context
        information and produces a fixed-dimensional state vector suitable for
        Q-learning.

        Args:
            terms: Current model specification [Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]]].
            runtime: TaskRuntime providing context masks [TaskRuntime].

        Returns:
            torch.Tensor: Encoded latent state vector Z [torch.Tensor].
        """
        k_ids, t_ids, g_ids, i_ids  = self._terms_to_id_tensors(terms)
        context_vector  = self.build_context_vector(runtime)
        z   = self.encoder(k_ids, t_ids, g_ids, i_ids, context_vector=context_vector) 
        return z
    
    def encode_many(self, batch_terms: Sequence[Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]]], task_ids: Sequence[int], runtimes_by_task: Dict[int, object],) -> torch.Tensor:
        """Encode a batch of specifications.

        This method allows encoding multiple model specifications simultaneously.
        Each specification is encoded using the runtime associated with its task identifier.

        Args:
            batch_terms: Batch of model specifications [Sequence[SpecificationTerm] | Sequence[tuple[int, int, int, int]]].
            task_ids: Task identifier for each specification [Sequence[int]].
            runtimes_by_task: Mapping from task id to TaskRuntime [Dict[int, TaskRuntime]].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, state_dim] [torch.Tensor].

        Raises:
            ValueError: If the batch lengths do not match.
        """
        if len(batch_terms) != len(task_ids):
            raise ValueError("batch_terms and task_ids must have the same length.")

        z_list = [self.encode_terms(terms=terms, runtime=runtimes_by_task[int(task_id)]) for terms, task_id in zip(batch_terms, task_ids)]
        
        if len(z_list) == 0:
            return torch.empty(0, self.state_dim, device=self.device)

        return torch.stack(z_list, dim=0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------    
    @torch.no_grad()
    def select_action(self, current_terms, action_manager, runtime, visited_state_keys, epsilon: float, boltzmann: bool = False, temperature: float = 1.0, ) -> tuple[int, torch.Tensor]:
        """Select an action from the current specification state.

        Action selection supports two exploration strategies:
        1. Epsilon-greedy: Random action with probability ε, greedy action otherwise.
        2. Boltzmann exploration: Samples actions according to a softmax distribution over Q-values.
        Invalid actions are masked before selection.

        Args:
            current_terms: Current specification [Sequence[SpecificationTerm]].
            action_manager: ActionManager instance [ActionManager].
            runtime: TaskRuntime instance [TaskRuntime].
            visited_state_keys: Previously visited states within the current episode [Sequence[str]].
            epsilon: Exploration probability [float].
            boltzmann: Whether Boltzmann exploration should be used [bool].
            temperature: Boltzmann temperature parameter [float].

        Returns:
            action_index: Selected global action index [int].
            z: Encoded latent state representation [torch.Tensor].

        """
        normalised_terms = self._normalise_terms(current_terms)
        valid_indices = action_manager.get_valid_action_indices(current_state=normalised_terms, visited_state_keys=visited_state_keys)
        z = self.encode_terms(normalised_terms, runtime=runtime)
        if not valid_indices:
            terminate_index = action_manager.catalogue_action_to_index[action_manager.catalogue_actions[0]]
            return terminate_index, z

        q = self.policy_net(z.unsqueeze(0))[0]

        if boltzmann:
            temp = max(float(temperature), 1e-8)
            valid_q = q[valid_indices]
            probs = torch.softmax(valid_q / temp, dim=0)
            local_choice = int(torch.multinomial(probs, num_samples=1).item())
            action_index = int(valid_indices[local_choice])
            return action_index, z

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_indices)), z

        q_masked = q.clone()
        invalid_mask = torch.ones_like(q_masked, dtype=torch.bool)
        invalid_mask[valid_indices] = False
        q_masked[invalid_mask] = torch.finfo(q.dtype).min
        action_index = int(torch.argmax(q_masked).item())        
        return action_index, z

    # ------------------------------------------------------------------
    # TD update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def polyak_update_(self) -> None:
        """Perform a soft update of the target network.

        Parameters are updated using Polyak averaging:
            θ_target ← (1 − τ) θ_target + τ θ_policy

        where τ is the target_soft_tau coefficient.
        """
        for target_param, source_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.lerp_(source_param.data, self.target_soft_tau)

    def update_from_batch(self, batch: list[Transition], indices: np.ndarray, is_w: np.ndarray, runtimes_by_task: Dict[int, object], replay_buffer,) -> dict:
        """Perform one Double-DQN optimization step.

        Training pipeline:
        1. Encode current states.
        2. Encode next states.
        3. Compute valid next-action masks.
        4. Compute Double-DQN targets.
        5. Compute loss.
        6. Apply importance-sampling weights.
        7. Backpropagate gradients.
        8. Clip gradients.
        9. Update replay priorities.
        10. Update target network using Polyak averaging.

        Args:
            batch: Batch of sampled transitions [list[Transition]].
            indices: Replay buffer indices [list[int]].
            is_w: Importance sampling weights [list[float]].
            runtimes_by_task: Mapping from task identifiers to runtimes [dict[int, TaskRuntime]].
            replay_buffer: Replay buffer instance [ReplayBuffer].

        Returns:
            dict:
                Training diagnostics containing:
                - loss: Loss value [float].
                - td_errors: TD errors for each transition [np.ndarray].
                - q_values_sa: Q-values for state-action pairs [np.ndarray].
                - target_values: Target values for each transition [np.ndarray].
                - grad_norm: Gradient norm [float].
                - param_update_magnitude: Parameter update magnitude [float].
        """
        task_ids        = [transition.task_id for transition in batch]
        state_terms     = [transition.state for transition in batch]
        next_terms      = [transition.next_state for transition in batch]
        action_indices  = [transition.action_index for transition in batch]
        rewards         = [transition.reward for transition in batch]
        dones           = [transition.done for transition in batch]

        z_t = self.encode_many(state_terms, task_ids, runtimes_by_task)
        next_z_t = self.encode_many(next_terms, task_ids, runtimes_by_task)

        a_t             = torch.tensor(action_indices, dtype=torch.long, device=self.device)
        r_t             = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d_t             = torch.tensor(dones, dtype=torch.float32, device=self.device)

        B = len(batch)
        valid_next = torch.zeros(B, self.num_actions, dtype=torch.bool, device=self.device) 

        for i, transition in enumerate(batch):
            if transition.done:
                continue
            next_valid_indices = getattr(transition, "next_valid_indices", None,)
            if next_valid_indices is None:
                raise ValueError("Transition is missing next_valid_indices. "
                    "Repeated-state masking requires storing next valid actions "
                    "during episode generation.")
            if len(next_valid_indices) > 0:
                valid_next[i, list(next_valid_indices)] = True

        has_valid_next = valid_next.any(dim=1)

        with torch.no_grad():
            q_next_policy = self.policy_net(next_z_t)
            q_next_policy = q_next_policy.masked_fill(~valid_next, torch.finfo(q_next_policy.dtype).min,)
            a_star = q_next_policy.argmax(dim=1)
            q_next_target = self.target_net(next_z_t).gather(1, a_star.unsqueeze(1)).squeeze(1)
            q_next_target = torch.where(has_valid_next, q_next_target, torch.zeros_like(q_next_target))
            target = r_t + (1.0 - d_t) * self.discount_factor * q_next_target

        # Snapshot parameters before update for magnitude tracking
        params_before = [
            p.data.detach().clone()
            for p in list(self.encoder.parameters()) + list(self.policy_net.parameters())
            if p.requires_grad
        ]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
            q_all = self.policy_net(z_t)
            q_s_a = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)
            per_sample_loss = torch.nn.functional.smooth_l1_loss(q_s_a, target, reduction="none",)
            w = torch.as_tensor(is_w, dtype=per_sample_loss.dtype, device=self.device)
            loss = (w * per_sample_loss).mean()

        self.scaler.scale(loss).backward()

        grad_norm = 0.0
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy_net.parameters()),
                    self.grad_clip_norm,
                ).item()
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            td_errors = (q_s_a - target).detach().abs().cpu().numpy()
            # Parameter update magnitude: mean L2 norm of weight deltas
            params_after = [
                p.data.detach()
                for p in list(self.encoder.parameters()) + list(self.policy_net.parameters())
                if p.requires_grad
            ]
            update_mag = float(np.mean([
                float(torch.norm(a - b).item())
                for a, b in zip(params_after, params_before)
            ])) if params_before else 0.0

        try:
            replay_buffer.update_priorities(indices, td_errors)
        except AttributeError:
            pass

        self.polyak_update_()

        return {
            "loss":                   float(loss.item()),
            "td_errors":              td_errors,
            "q_values_sa":            q_s_a.detach().cpu().numpy(),
            "target_values":          target.detach().cpu().numpy(),
            "grad_norm":              grad_norm,
            "param_update_magnitude": update_mag,
        }
    
    def summary(self) -> dict:
        """Return a compact summary of the agent configuration.

        Returns:
            dict:
                Dictionary containing:
                - device: Device used for training [str].
                - encoder_kind: Type of encoder used [str].
                - state_dim: Dimension of the state space [int].
                - num_actions: Number of possible actions [int].
                - learning_rate: Learning rate used for training [float].
                - discount_factor: Discount factor used for training [float].
                - grad_clip_norm: Gradient clip norm used for training [float].
                - target_soft_tau: Target soft tau used for training [float].
                - use_amp: Whether to use automatic mixed precision [bool].
        """
        return {
            "device": str(self.device),
            "num_actions": self.num_actions,
            "encoder_kind": self.encoder_kind,
            "state_dim": self.state_dim,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "grad_clip_norm": self.grad_clip_norm,
            "target_soft_tau": self.target_soft_tau,
            "use_amp": self.use_amp,
        }