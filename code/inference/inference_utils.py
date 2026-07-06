"""
inference/inference_utils.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: shared utilities for inference execution across zero-shot and few-shot paradigms.
"""
from __future__ import annotations

from trainer.trainer import Trainer

__all__ = ["generate_episode_with_strategy"]


def generate_episode_with_strategy(
    trainer: Trainer, 
    runtime, 
    strategy: str, 
    epsilon: float = 0.10, 
    temperature: float = 1.0, 
    top_k: int = 5
):
    """
    Generates an episode using the specified strategy.
    
    Supported Strategies:
        - 'greedy'
        - 'stochastic'
        - 'boltzmann'
        - 'topk'
    """
    strategy = strategy.lower()
    if strategy == "greedy":
        return trainer.generate_episode(runtime=runtime, epsilon=0.0, boltzmann=False)
    if strategy == "stochastic":
        return trainer.generate_episode(runtime=runtime, epsilon=epsilon, boltzmann=False)
    if strategy == "boltzmann":
        return trainer.generate_episode(runtime=runtime, epsilon=0.0, boltzmann=True, temperature=temperature)
    if strategy == "topk":
        return trainer.generate_episode_topk(runtime=runtime, k=top_k, temperature=temperature)       
    raise ValueError(f"Unknown strategy '{strategy}'. Expected greedy, stochastic, boltzmann, or topk.")
