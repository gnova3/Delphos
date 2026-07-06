"""
agent/delphos_dqn.py
: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Pure DQN Reinforcement learning agent using one-hot encoded state.

DelphosDQN is a Deep Q-Network (DQN) agent designed to
automate the specification of discrete choice models.
Unlike the standard Delphos agent, this agent flattens the 
specification and uses a one-hot encoding of the tuples.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.dqn import QNetwork
from agent.encoder import ZStateConfig
from agent.replay_buffer import Transition
from mdp.runtime import Runtime

__all__ = ["DelphosDQN", "OneHotStateEncoder"]


class OneHotStateEncoder(nn.Module):
    """
    Encodes the specification tensor into a flattened one-hot vector.
    Each row (attribute) in the specification has a transform_id, taste_id, and covariate_id.
    This encoder one-hot encodes these IDs and concatenates them, then flattens across all K attributes.
    """
    def __init__(self, cfg: ZStateConfig):
        super().__init__()
        self.cfg = cfg
        # For each attribute, we one-hot encode transform, taste, and covariate.
        self.state_dim = cfg.K * ((cfg.T + 1) + (cfg.G + 1) + (cfg.C + 1))
        
        # Optional projection head
        if cfg.head_flag:
            self.head = nn.Sequential(
                nn.Linear(self.state_dim + cfg.context_dim, cfg.d_state),
                nn.ReLU()
            )
            self.out_dim = cfg.d_state
        else:
            self.out_dim = self.state_dim + cfg.context_dim
            self.head = None

    def forward(self, attribute_ids: torch.LongTensor, transform_ids: torch.LongTensor, taste_ids: torch.LongTensor, covariate_ids: torch.LongTensor, context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        is_single = transform_ids.dim() == 1
        if is_single:
            transform_ids = transform_ids.unsqueeze(0)
            taste_ids = taste_ids.unsqueeze(0)
            covariate_ids = covariate_ids.unsqueeze(0)

        # One-hot encode each component
        # Shapes will be [B, K, num_classes]
        t_onehot = F.one_hot(transform_ids, num_classes=self.cfg.T + 1).float()
        g_onehot = F.one_hot(taste_ids, num_classes=self.cfg.G + 1).float()
        c_onehot = F.one_hot(covariate_ids, num_classes=self.cfg.C + 1).float()

        # Concatenate along the last dimension: [B, K, (T+1) + (G+1) + (C+1)]
        x = torch.cat([t_onehot, g_onehot, c_onehot], dim=-1)

        # Flatten the K dimension: [B, K * ((T+1) + (G+1) + (C+1))]
        B = x.shape[0]
        x = x.view(B, -1)

        # Append context vector if necessary
        if self.cfg.context_dim > 0 and context_vector is not None:
            if context_vector.dim() == 1:
                context_vector = context_vector.unsqueeze(0).expand(B, -1)
            x = torch.cat([x, context_vector], dim=-1)

        if self.head is not None:
            x = self.head(x)

        if is_single:
            x = x.squeeze(0)

        return x


class DelphosDQN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        z_cfg: ZStateConfig,
        encoder_kind: str = "onehot",
        learning_rate: float = 1e-3,
        discount_factor: float = 0.90,
        target_soft_tau: float = 5e-3,
        grad_clip_norm: Optional[float] = 1.0,
        device: str = "cpu",
        use_amp: bool = False,) -> None:

        super().__init__()

        self.device = torch.device(device)
        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.target_soft_tau = float(target_soft_tau)
        self.grad_clip_norm = grad_clip_norm
        self.encoder_kind = str(encoder_kind).lower()

        # --- Shared representation ---
        self.z_cfg = z_cfg
        self.encoder = OneHotStateEncoder(self.z_cfg)
        self.encoder.to(self.device)

        # --- Q networks ---
        self.state_dim = self.encoder.out_dim
        
        # Policy network
        self.policy_net = QNetwork(input_size=self.state_dim, output_size=self.num_actions, hidden_layers=(256, 256)).to(self.device)

        # Target network
        self.target_net = QNetwork(input_size=self.state_dim, output_size=self.num_actions, hidden_layers=(256, 256)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- Optimizer ---
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.policy_net.parameters()), lr=self.learning_rate)
        self.autocast_device = ("cuda" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu")
        self.use_amp = (bool(use_amp) and self.autocast_device == "cuda")
        self.scaler = torch.amp.GradScaler(self.autocast_device, enabled=self.use_amp)
        self.mode = "full_training"
        self.full_training_mode()


    def encode_specification(self, specification: torch.LongTensor, runtime: Runtime,) -> torch.Tensor:
        """
        Encode a specification tensor into a one-hot state representation.
        """

        runtime.specification.validate(specification)
        specification = specification.to(self.device)

        attribute_ids = specification[:, 0]
        transform_ids = specification[:, 1]
        taste_ids = specification[:, 2]
        covariate_ids = specification[:, 3]
        context_vector = runtime.context_vector.to(self.device)
        z = self.encoder(attribute_ids, transform_ids, taste_ids, covariate_ids, context_vector=context_vector)
        return z

    def encode_batch(self, specifications: list[torch.LongTensor], task_ids: list[int], runtimes: dict[int, Runtime],) -> torch.Tensor:
        """
        Encode a batch of specifications.
        """

        if len(specifications) != len(task_ids):
            raise ValueError("specifications and task_ids must have identical length.")
        
        z_list = []
        for specification, task_id in zip(specifications, task_ids,):
            runtime = runtimes[int(task_id)]
            z = self.encode_specification(specification=specification,runtime=runtime)
            z_list.append(z)

        if len(z_list) == 0:
            return torch.empty(0, self.state_dim, device=self.device)

        return torch.stack(z_list,dim=0)

    @torch.no_grad()
    def q_values(self, specification: torch.LongTensor, runtime: Runtime) -> torch.Tensor:
        """
        Compute Q-values for all catalogue actions.
        """
        z = self.encode_specification(specification=specification, runtime=runtime)
        q_values = self.policy_net(z.unsqueeze(0))[0]
        return q_values

    @torch.no_grad()
    def eval_q(self, specification, runtime, action_idx):
        q = self.q_values(specification, runtime)
        return float(q[action_idx].item())

    @torch.no_grad()
    def select_action(
        self,
        specification: torch.LongTensor,
        runtime: Runtime,
        visited_specifications: set[str],
        epsilon: float,
        boltzmann: bool = False,
        temperature: float = 1.0,
    ) -> tuple[int, torch.Tensor]:
        """
        Select an action from the current specification.
        """

        valid_indices = runtime.action_space.get_valid_action_indices(specification, visited_specifications)
        z = self.encode_specification(specification, runtime)

        # No valid actions
        if len(valid_indices) == 0:
            return runtime.action_space.terminate_action_index, z

        q = self.policy_net(z.unsqueeze(0))[0]
        
        # Boltzmann exploration
        if boltzmann:
            temperature = max(float(temperature), 1e-8)
            valid_q = q[valid_indices]
            probabilities = torch.softmax(valid_q / temperature, dim=0)
            local_idx = int(torch.multinomial(probabilities, num_samples=1).item())
            action_idx = int(valid_indices[local_idx])
            return action_idx, z

        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action_idx = int(np.random.choice(valid_indices))
            return action_idx, z

        # Greedy action
        q_masked = q.clone()
        invalid_mask = torch.ones_like(q_masked, dtype=torch.bool)
        invalid_mask[valid_indices] = False
        q_masked[invalid_mask] = torch.finfo(q.dtype).min
        action_idx = int(torch.argmax(q_masked).item())
        return action_idx, z

    @torch.no_grad()
    def greedy_action(self, specification: torch.LongTensor, runtime: Runtime, visited_specifications: set[str],) -> tuple[int, torch.Tensor]:
        return self.select_action(specification,runtime,visited_specifications,0.0,False,)

    @torch.no_grad()
    def random_action(self, specification: torch.LongTensor,runtime: Runtime, visited_specifications: set[str],) -> tuple[int, torch.Tensor]:
        valid_indices = runtime.action_space.get_valid_action_indices(specification, visited_specifications)
        z = self.encode_specification(specification,runtime)
        if len(valid_indices) == 0:
            return runtime.action_space.terminate_action_index, z
        return int(np.random.choice(valid_indices)), z

    @torch.no_grad()
    def topk_action(self,specification: torch.LongTensor,runtime: Runtime,visited_specifications: set[str],k: int = 5,temperature: float | None = None) -> tuple[int, torch.Tensor]:
        valid_indices = runtime.action_space.get_valid_action_indices(specification,visited_specifications)
        z = self.encode_specification(specification,runtime)

        if len(valid_indices) == 0:
            return runtime.action_space.terminate_action_index, z

        q = self.policy_net(z.unsqueeze(0))[0]
        valid_q = q[valid_indices]
        k = min(int(k), len(valid_indices))
        topk_values, topk_indices = torch.topk(valid_q,k=k,largest=True,)
        candidate_q = topk_values
        candidate_actions = [valid_indices[int(idx)] for idx in topk_indices.cpu().tolist()]
    
        if temperature is None:
            sampled_idx = int(torch.randint(0,len(candidate_actions),(1,),device=candidate_q.device,).item())
            return int(candidate_actions[sampled_idx]), z
    
        temperature = max(float(temperature), 1e-8)
        probabilities = torch.softmax(candidate_q / temperature,dim=0)
        sampled_idx = int(torch.multinomial(probabilities, num_samples=1).item())
        return int(candidate_actions[sampled_idx]), z

    @torch.no_grad()
    def polyak_update(self) -> None:
        for target_param, source_param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.lerp_(source_param.data,self.target_soft_tau)

    def update_from_batch(self, batch: list[Transition], indices: np.ndarray, is_w: np.ndarray, runtimes: dict[int, Runtime], replay_buffer) -> dict:

        if len(batch) == 0:
            raise ValueError("Cannot update using an empty batch.")

        # Unpack batch
        task_ids = [transition.task_id for transition in batch]
        states = [transition.state for transition in batch]
        next_states = [transition.next_state for transition in batch]
        actions = [transition.action_index for transition in batch]
        rewards = [transition.reward for transition in batch]
        dones = [transition.done for transition in batch]

        # Encode states
        z_t = self.encode_batch(specifications=states,task_ids=task_ids,runtimes=runtimes,)
        next_z_t = self.encode_batch(specifications=next_states,task_ids=task_ids,runtimes=runtimes,)

        # Convert to tensors
        action_tensor = torch.tensor(actions, dtype=torch.long,device=self.device,)
        reward_tensor = torch.tensor(rewards,dtype=torch.float32,device=self.device,)
        done_tensor = torch.tensor(dones,dtype=torch.float32,device=self.device,)
        batch_size = len(batch)

        # Valid next-action mask
        valid_next_mask = torch.zeros(batch_size,self.num_actions,dtype=torch.bool,device=self.device,)
        for row_idx, transition in enumerate(batch):
            if transition.done:
                continue
            if len(transition.next_valid_indices) > 0:
                valid_next_mask[row_idx,transition.next_valid_indices,] = True

        has_valid_next = valid_next_mask.any(dim=1)

        # Double-DQN target
        with torch.no_grad():
            q_next_policy = self.policy_net(next_z_t)
            q_next_policy = q_next_policy.masked_fill(~valid_next_mask,torch.finfo(q_next_policy.dtype).min,)
            next_actions = q_next_policy.argmax(dim=1)
            q_next_target = (self.target_net(next_z_t).gather(1,next_actions.unsqueeze(1),)).squeeze(1)
            q_next_target = torch.where(has_valid_next,q_next_target,torch.zeros_like(q_next_target,),)
            target_values = (reward_tensor + (1.0 - done_tensor) * self.discount_factor * q_next_target)

        # Snapshot parameters
        parameters_before = [parameter.detach().clone() for parameter in (list(self.encoder.parameters())+ list(self.policy_net.parameters())) if parameter.requires_grad]
        
        # Compute Loss
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.autocast_device,enabled=self.use_amp,):
            q_values = self.policy_net(z_t)
            q_sa = (q_values.gather(1,action_tensor.unsqueeze(1),).squeeze(1))
            per_sample_loss = (torch.nn.functional.smooth_l1_loss(q_sa, target_values,reduction="none",))
            weights = torch.as_tensor(is_w,dtype=per_sample_loss.dtype,device=self.device,)
            loss = (weights * per_sample_loss).mean()

        # Backpropagation
        self.scaler.scale(loss).backward()

        # Gradient Clipping
        grad_norm = 0.0
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+ list(self.policy_net.parameters()),self.grad_clip_norm,).item())

        # Optimizer Step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Diagnostics
        with torch.no_grad():
            td_errors = (q_sa - target_values).abs().cpu().numpy()
            parameters_after = [parameter.detach() for parameter in (list(self.encoder.parameters())+ list(self.policy_net.parameters())) if parameter.requires_grad]
            if len(parameters_before) == 0:
                   update_magnitude = 0.0
            else:
                update_magnitude = float(np.mean([torch.norm(after - before).item() for before, after in zip(parameters_before, parameters_after,)]))

        # Priority replay update
        try:
            replay_buffer.update_priorities(indices, td_errors,)
        except AttributeError:
            pass

        # Target update
        self.polyak_update()

        return {
            "loss": float(loss.item()),
            "td_errors": td_errors,
            "q_values_sa": q_sa.detach().cpu().numpy(),
            "target_values": target_values.detach().cpu().numpy(),
            "grad_norm": grad_norm,
            "param_update_magnitude": update_magnitude,
        }

    # =====================================================
    # Checkpointing
    # =====================================================
    def checkpoint_metadata(self):
        return {
            "agent": "DelphosDQN",
            "encoder_kind": self.encoder_kind,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "z_cfg": vars(self.z_cfg),
        }

    def save_checkpoint(self, path: str, save_mode: str = "full", extra_metadata: Optional[dict] = None) -> None:
        metadata = {**self.checkpoint_metadata(), **self.summary()}
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        checkpoint = {"metadata": metadata,"rng": {"numpy": np.random.get_state(),"torch": torch.get_rng_state(),},}
        if torch.cuda.is_available():
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        if save_mode in {"full", "encoder"}:
            checkpoint["encoder"] = self.encoder.state_dict()
        if save_mode in {"full", "policy"}:
            checkpoint["policy_net"] = self.policy_net.state_dict()
        if save_mode == "full":
            checkpoint["target_net"] = self.target_net.state_dict()
            checkpoint["optimizer"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
    
    def _load_partial_state_dict(self, module: nn.Module, state_dict: dict) -> tuple[int, int]:
        current = module.state_dict()
        loaded = 0
        skipped = 0
        compatible = {}
        for key, tensor in state_dict.items():
            if key not in current:
                skipped += 1
                continue
            if current[key].shape != tensor.shape:
                skipped += 1
                continue
            compatible[key] = tensor
            loaded += 1
        current.update(compatible)
        module.load_state_dict(current)
        return loaded, skipped

    def load_checkpoint(self, path: str, load_mode: str = "resume",) -> dict:
        checkpoint = torch.load(path,map_location=self.device,weights_only=False)
        metadata = checkpoint.get("metadata", {})

        if load_mode == "resume":
            if "encoder" in checkpoint:
                self.encoder.load_state_dict(checkpoint["encoder"])
            if "policy_net" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["policy_net"])
            if "target_net" in checkpoint:
                self.target_net.load_state_dict(checkpoint["target_net"])
            if "optimizer" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception:
                    pass
            if "rng" in checkpoint:
                rng = checkpoint["rng"]
                if "numpy" in rng:
                    np.random.set_state(rng["numpy"])
                if "torch" in rng:
                    torch.set_rng_state(rng["torch"])
            if "cuda_rng" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            saved_mode = metadata.get("mode","full_training")
            try:
                self.set_mode(saved_mode)
            except Exception:
                self.full_training_mode()

            return metadata

        if load_mode == "encoder":
            if "encoder" not in checkpoint:
                raise ValueError("Checkpoint does not contain encoder.")
            loaded, skipped = self._load_partial_state_dict(self.encoder,checkpoint["encoder"])
            metadata["loaded_parameters"] = loaded
            metadata["skipped_parameters"] = skipped
            return metadata

        if load_mode == "policy":
            if "policy_net" not in checkpoint:
                raise ValueError("Checkpoint does not contain policy.")
            loaded, skipped = self._load_partial_state_dict(self.policy_net,checkpoint["policy_net"])
            metadata["loaded_parameters"] = loaded
            metadata["skipped_parameters"] = skipped
            return metadata

        if load_mode == "transfer":
            loaded_encoder = 0
            loaded_policy = 0
            if "encoder" in checkpoint:
                loaded_encoder, _ = self._load_partial_state_dict(self.encoder,checkpoint["encoder"])
            if "policy_net" in checkpoint:
                loaded_policy, _ = self._load_partial_state_dict(self.policy_net,checkpoint["policy_net"])
            metadata["loaded_encoder"] = loaded_encoder
            metadata["loaded_policy"] = loaded_policy
            return metadata
        raise ValueError(f"Unknown load_mode '{load_mode}'.")

    # =====================================================
    # Training and Inference modes
    # =====================================================

    def rebuild_optimizer(self) -> None:
        trainable_parameters = []
        trainable_parameters.extend(parameter for parameter in self.encoder.parameters() if parameter.requires_grad)
        trainable_parameters.extend(parameter for parameter in self.policy_net.parameters() if parameter.requires_grad)
        if len(trainable_parameters) == 0:
            raise ValueError("No trainable parameters available for optimizer construction.")

        self.optimizer = optim.Adam(trainable_parameters,lr=self.learning_rate,)

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.encoder.train()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def freeze_policy(self) -> None:
        self.policy_net.eval()
        for parameter in self.policy_net.parameters():
            parameter.requires_grad = False

    def unfreeze_policy(self) -> None:
        self.policy_net.train()
        for parameter in self.policy_net.parameters():
            parameter.requires_grad = True

    def set_mode(self, mode: str) -> None:
        mode = mode.lower()

        if mode == "inference":
            self.inference_mode()
        elif mode == "policy_only":
            self.fine_tune_policy_mode()
        elif mode == "encoder_only":
            self.fine_tune_encoder_mode()
        elif mode == "fine_tune_all":
            self.fine_tune_policy_and_encoder_mode()
        elif mode == "full_training":
            self.full_training_mode()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: inference, policy_only, encoder_only, fine_tune_all, full_training.")

    def inference_mode(self) -> None:
        self.mode = "inference"
        self.freeze_encoder()
        self.freeze_policy()
        self.target_net.eval()

    def fine_tune_policy_mode(self) -> None:
        self.mode = "policy_only"
        self.freeze_encoder()
        self.unfreeze_policy()
        self.target_net.eval()
        self.rebuild_optimizer()

    def fine_tune_encoder_mode(self) -> None:
        self.mode = "encoder_only"
        self.unfreeze_encoder()
        self.freeze_policy()
        self.target_net.eval()
        self.rebuild_optimizer()

    def fine_tune_policy_and_encoder_mode(self) -> None:
        self.mode = "fine_tune_all"
        self.unfreeze_encoder()
        self.unfreeze_policy()
        self.target_net.eval()
        self.rebuild_optimizer()

    def full_training_mode(self) -> None:
        self.mode = "full_training"
        self.unfreeze_encoder()
        self.unfreeze_policy()
        self.policy_net.train()
        self.target_net.train()
        self.encoder.train()
        self.rebuild_optimizer()

    @torch.no_grad()
    def count_parameters(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        encoder_trainable = sum(parameter.numel() for parameter in self.encoder.parameters() if parameter.requires_grad)
        policy_trainable = sum(parameter.numel() for parameter in self.policy_net.parameters() if parameter.requires_grad)

        return {"total": int(total), "trainable": int(trainable), 
                "encoder_trainable": int(encoder_trainable), "policy_trainable": int(policy_trainable),}

    @torch.no_grad()
    def summary(self) -> dict:
        return {
            "agent": "DelphosDQN",
            "mode": self.mode,
            "encoder_kind": self.encoder_kind,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "z_cfg": vars(self.z_cfg),
            "encoder_class": type(self.encoder).__name__,
            "policy_class": type(self.policy_net).__name__,
            "target_class": type(self.target_net).__name__,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "target_soft_tau": self.target_soft_tau,
            "grad_clip_norm": self.grad_clip_norm,            
            "device": str(self.device),
            "use_amp": self.use_amp,
            "parameters": self.count_parameters(),
            "encoder_frozen": not any(p.requires_grad for p in self.encoder.parameters()),
            "policy_frozen": not any(p.requires_grad for p in self.policy_net.parameters()),
        }

    def __repr__(self):
        return (
            f"DelphosDQN("
            f"encoder={self.encoder_kind}, "
            f"actions={self.num_actions}, "
            f"state_dim={self.state_dim}, "
            f"mode={self.mode}"
            f")"
        )
