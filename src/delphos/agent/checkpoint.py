from __future__ import annotations

from pathlib import Path

import torch

from delphos.agent.encoder import ZStateConfig
from delphos.agent.inference_agent import DelphosAgent


def load_agent_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    strict: bool = True,
) -> DelphosAgent:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    payload = checkpoint.get("agent_checkpoint", checkpoint)
    metadata = payload.get("metadata", {})
    if not metadata:
        raise ValueError("Checkpoint does not contain agent metadata.")

    z_cfg = ZStateConfig(**metadata["z_cfg"])
    agent = DelphosAgent(
        num_actions=int(metadata["num_actions"]),
        z_cfg=z_cfg,
        encoder_kind=metadata.get("encoder_kind", "deepset"),
        discount_factor=float(metadata.get("discount_factor", 0.90)),
        device=device,
    )
    agent.load_checkpoint_payload(payload, strict=strict)
    return agent
