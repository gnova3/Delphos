from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def resolve_resume_path(path: str) -> Optional[str]:
    """Resolve a checkpoint path or directory to a specific valid .pt file.

    Args:
        path (str): The provided path string.

    Returns:
        Optional[str]: The resolved file path, or None if not found.
    """
    path_obj = Path(path)

    if path_obj.is_file():
        return str(path_obj)

    if path_obj.is_dir():
        candidates = [
            fp for fp in path_obj.iterdir()
            if fp.is_file() and fp.name.startswith("checkpoint_") and fp.suffix == ".pt"
        ]
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
            return str(candidates[-1])

        for fallback in ("delphos_agent.pt", "checkpoint_latest.pt"):
            fp = path_obj / fallback
            if fp.is_file():
                return str(fp)

    return None


def save_checkpoint(
    path: str,
    *,
    agent,
    episode_count: int,
    learn_steps: int,
    epsilon: float,
    extra_metadata: Optional[dict] = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path (str): The file path to save the checkpoint to.
        agent: The DelphosAgent instance.
        episode_count (int): The current training episode count.
        learn_steps (int): The current learning step count.
        epsilon (float): The current epsilon value.
        extra_metadata (Optional[dict], optional): Additional metadata to save. Defaults to None.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "policy_state": agent.policy_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "encoder_state": agent.encoder.state_dict(),
        "term_encoder_state": agent.term_encoder.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "episode_count": int(episode_count),
        "learn_steps": int(learn_steps),
        "epsilon": float(epsilon),
        "metadata": dict(extra_metadata or {}),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, str(path_obj))


def load_state_dict_safely(
    model: nn.Module,
    state_dict: dict,
    *,
    reset_output_if_mismatch: bool = True,
) -> dict[str, list[str]]:
    """Load a state dictionary safely into a PyTorch module.

    Handles shape mismatches gracefully and records skipped keys.

    Args:
        model (nn.Module): The PyTorch module to load weights into.
        state_dict (dict): The source state dictionary.
        reset_output_if_mismatch (bool, optional): If True, re-initialize the last linear layer
            if shape mismatches occur. Defaults to True.

    Returns:
        dict[str, list[str]]: A report mapping 'loaded', 'skipped_shape', 'skipped_missing', 
            'missing_after', and 'unexpected_after' keys to lists of parameter names.
    """
    current = model.state_dict()
    loaded, skipped_shape, skipped_missing = [], [], []
    copy_dict = {}

    for key, value in state_dict.items():
        if key not in current:
            skipped_missing.append(key)
            continue
        if current[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        copy_dict[key] = value
        loaded.append(key)

    missing, unexpected = model.load_state_dict(copy_dict, strict=False)

    if reset_output_if_mismatch and skipped_shape:
        try:
            last_linear = None
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.kaiming_uniform_(
                    last_linear.weight,
                    a=0.0,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)
        except Exception:
            pass

    return {
        "loaded": loaded,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "missing_after": list(missing),
        "unexpected_after": list(unexpected),
    }


def load_agent_checkpoint(
    path: str,
    *,
    agent,
    optimizer=None,
    mode: str = "resume",
    strict: bool = False,
    load_target: bool = True,
    reset_output_if_mismatch: bool = True,
) -> dict:
    """Load an agent from a saved checkpoint safely.

    Supports resuming training, full fine-tuning, or policy-only loading.

    Args:
        path (str): Path to the checkpoint file.
        agent: The DelphosAgent instance to populate.
        optimizer: The PyTorch optimizer (optional).
        mode (str, optional): The loading mode ("resume", "finetune", "policy_only"). Defaults to "resume".
        strict (bool, optional): Whether to enforce strict dictionary loading. Defaults to False.
        load_target (bool, optional): Whether to load the target network. Defaults to True.
        reset_output_if_mismatch (bool, optional): Whether to reset output layers on mismatch. Defaults to True.

    Returns:
        dict: A dictionary containing loaded `episode_count`, `learn_steps`, `epsilon`,
            `metadata`, and a `report` of loaded/skipped layers.
    """
    if mode not in {"resume", "finetune", "policy_only"}:
        raise ValueError(f"Unknown load mode: {mode}")

    ckpt = torch.load(path, map_location=agent.device, weights_only=False)

    policy_state = ckpt.get("policy_state", ckpt.get("state_dict", None))
    target_state = ckpt.get("target_state", None)
    encoder_state = ckpt.get("encoder_state", None)
    term_encoder_state = ckpt.get("term_encoder_state", None)
    optimizer_state = ckpt.get("optimizer_state", None)

    if policy_state is None:
        raise ValueError("Checkpoint missing policy_state (or state_dict).")

    report = {}

    if mode != "policy_only":
        if encoder_state is not None:
            report["encoder"] = load_state_dict_safely(
                agent.encoder,
                encoder_state,
                reset_output_if_mismatch=False,
            )

        if term_encoder_state is not None:
            report["term_encoder"] = load_state_dict_safely(
                agent.term_encoder,
                term_encoder_state,
                reset_output_if_mismatch=False,
            )

    if strict:
        agent.policy_net.load_state_dict(policy_state, strict=True)
        if load_target and target_state is not None:
            agent.target_net.load_state_dict(target_state, strict=True)
        elif load_target:
            agent.target_net.load_state_dict(agent.policy_net.state_dict(), strict=False)
    else:
        report["policy"] = load_state_dict_safely(
            agent.policy_net,
            policy_state,
            reset_output_if_mismatch=reset_output_if_mismatch,
        )
        if load_target and target_state is not None:
            report["target"] = load_state_dict_safely(
                agent.target_net,
                target_state,
                reset_output_if_mismatch=reset_output_if_mismatch,
            )
        elif load_target:
            agent.target_net.load_state_dict(agent.policy_net.state_dict(), strict=False)

    if mode == "resume" and optimizer is not None and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            pass

    if mode == "resume":
        try:
            rng = ckpt.get("rng", {})
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "torch_cpu" in rng and rng["torch_cpu"] is not None:
                torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in rng and rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass

    return {
        "episode_count": int(ckpt.get("episode_count", 0)),
        "learn_steps": int(ckpt.get("learn_steps", 0)),
        "epsilon": float(ckpt.get("epsilon", 1.0)),
        "metadata": ckpt.get("metadata", {}),
        "report": report,
    }