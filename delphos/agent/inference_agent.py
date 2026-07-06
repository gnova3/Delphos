from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from delphos.agent.encoder import (
    DeepSetEncoder,
    InteractionAwareEncoder,
    TermEncoder,
    ZStateConfig,
)
from delphos.agent.q_network import QNetwork


class DelphosAgent(nn.Module):
    """Inference-only Delphos policy used to propose model specifications."""

    def __init__(
        self,
        num_actions: int,
        z_cfg: ZStateConfig,
        encoder_kind: str = "deepset",
        discount_factor: float = 0.90,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_actions = int(num_actions)
        self.discount_factor = float(discount_factor)
        self.encoder_kind = str(encoder_kind).lower()
        self.z_cfg = z_cfg

        self.term_encoder = TermEncoder(self.z_cfg)
        if self.encoder_kind == "deepset":
            self.encoder = DeepSetEncoder(self.z_cfg, self.term_encoder)
        elif self.encoder_kind in {"interaction", "attention", "interaction_aware"}:
            self.encoder = InteractionAwareEncoder(self.z_cfg, self.term_encoder)
        else:
            raise ValueError(f"Unknown encoder_kind '{encoder_kind}'")

        self.state_dim = (
            self.z_cfg.d_state
            if self.z_cfg.head_flag
            else (self.z_cfg.d_term + self.z_cfg.context_dim)
        )
        self.policy_net = QNetwork(
            input_size=self.state_dim,
            output_size=self.num_actions,
            hidden_layers=(256, 256),
        )
        self.target_net = QNetwork(
            input_size=self.state_dim,
            output_size=self.num_actions,
            hidden_layers=(256, 256),
        )
        self.to(self.device)
        self.inference_mode()

    @torch.no_grad()
    def load_checkpoint_payload(self, payload: dict, *, strict: bool = True) -> dict:
        if "encoder" not in payload:
            raise ValueError("Checkpoint payload is missing encoder weights.")
        if "policy_net" not in payload:
            raise ValueError("Checkpoint payload is missing policy_net weights.")

        encoder_result = self.encoder.load_state_dict(payload["encoder"], strict=strict)
        policy_result = self.policy_net.load_state_dict(payload["policy_net"], strict=strict)
        target_result = None
        if "target_net" in payload:
            target_result = self.target_net.load_state_dict(payload["target_net"], strict=strict)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)
        self.inference_mode()
        return {
            "encoder": encoder_result,
            "policy_net": policy_result,
            "target_net": target_result,
            "metadata": payload.get("metadata", {}),
        }

    def inference_mode(self) -> None:
        self.mode = "inference"
        self.encoder.eval()
        self.policy_net.eval()
        self.target_net.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def encode_specification(self, specification: torch.LongTensor, runtime) -> torch.Tensor:
        runtime.specification.validate(specification)
        specification = specification.to(self.device)
        attribute_ids = specification[:, 0]
        transform_ids = specification[:, 1]
        taste_ids = specification[:, 2]
        covariate_ids = specification[:, 3]
        context_vector = runtime.context_vector.to(self.device)
        return self.encoder(
            attribute_ids,
            transform_ids,
            taste_ids,
            covariate_ids,
            context_vector=context_vector,
        )

    @torch.no_grad()
    def q_values(self, specification: torch.LongTensor, runtime) -> torch.Tensor:
        z = self.encode_specification(specification=specification, runtime=runtime)
        return self.policy_net(z.unsqueeze(0))[0]

    @torch.no_grad()
    def select_action(
        self,
        specification: torch.LongTensor,
        runtime,
        visited_specifications: set[str],
        epsilon: float,
        boltzmann: bool = False,
        temperature: float = 1.0,
    ) -> tuple[int, torch.Tensor]:
        valid_indices = runtime.action_space.get_valid_action_indices(
            specification,
            visited_specifications,
        )
        z = self.encode_specification(specification, runtime)
        if len(valid_indices) == 0:
            return runtime.action_space.terminate_action_index, z

        q = self.policy_net(z.unsqueeze(0))[0]
        if boltzmann:
            temperature = max(float(temperature), 1e-8)
            valid_q = q[valid_indices]
            probabilities = torch.softmax(valid_q / temperature, dim=0)
            local_idx = int(torch.multinomial(probabilities, num_samples=1).item())
            return int(valid_indices[local_idx]), z

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_indices)), z

        q_masked = q.clone()
        invalid_mask = torch.ones_like(q_masked, dtype=torch.bool)
        invalid_mask[valid_indices] = False
        q_masked[invalid_mask] = torch.finfo(q.dtype).min
        return int(torch.argmax(q_masked).item()), z

    @torch.no_grad()
    def topk_action(
        self,
        specification: torch.LongTensor,
        runtime,
        visited_specifications: set[str],
        k: int = 5,
        temperature: Optional[float] = None,
    ) -> tuple[int, torch.Tensor]:
        valid_indices = runtime.action_space.get_valid_action_indices(
            specification,
            visited_specifications,
        )
        z = self.encode_specification(specification, runtime)
        if len(valid_indices) == 0:
            return runtime.action_space.terminate_action_index, z

        q = self.policy_net(z.unsqueeze(0))[0]
        valid_q = q[valid_indices]
        k = min(int(k), len(valid_indices))
        topk_values, topk_indices = torch.topk(valid_q, k=k, largest=True)
        candidate_actions = [valid_indices[int(idx)] for idx in topk_indices.cpu().tolist()]

        if temperature is None:
            sampled_idx = int(torch.randint(0, len(candidate_actions), (1,), device=q.device).item())
            return int(candidate_actions[sampled_idx]), z

        temperature = max(float(temperature), 1e-8)
        probabilities = torch.softmax(topk_values / temperature, dim=0)
        sampled_idx = int(torch.multinomial(probabilities, num_samples=1).item())
        return int(candidate_actions[sampled_idx]), z

    @torch.no_grad()
    def summary(self) -> dict:
        return {
            "agent": "DelphosAgent",
            "mode": self.mode,
            "encoder_kind": self.encoder_kind,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "z_cfg": vars(self.z_cfg),
            "device": str(self.device),
        }
