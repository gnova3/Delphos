"""
inference/inference_zero_shot_learning.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: zero-shot inference learning utilities for Delphos.
"""
from __future__ import annotations

import copy
import pandas as pd
from trainer.trainer import Trainer

from inference.inference_utils import generate_episode_with_strategy

__all__ = ["run_zero_shot_inference"]


def run_zero_shot_inference(
    trainer: Trainer,
    task,
    n_models: int = 100,
    max_attempts: int = 500,
    strategy: str = "boltzmann",
    epsilon: float = 0.10,
    temperature: float = 1.0,
    top_k: int = 5,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Runs zero-shot inference to collect up to `n_models` unique specifications.
    The agent is not trained during this process.
    """
    runtime = trainer.add_task(task=task, overwrite=overwrite)
    
    agent_backup = copy.deepcopy(trainer.agent.state_dict())
    local_buffer = copy.deepcopy(trainer.replay_buffer)

    unique_models = {}
    attempts = 0
    valid_models_count = 0

    try:
        trainer.agent.inference_mode()
        
        while valid_models_count < n_models and attempts < max_attempts:
            episode = generate_episode_with_strategy(
                trainer=trainer, 
                runtime=runtime, 
                strategy=strategy, 
                epsilon=epsilon, 
                temperature=temperature, 
                top_k=top_k
            )
            
            spec_key = episode.specification_key
            if spec_key not in unique_models:
                reward = float(episode.reward)
                unique_models[spec_key] = {
                    "task_id": int(runtime.task.id),
                    "task_name": str(runtime.task.name),
                    "specification_key": spec_key,
                    "reward": reward,
                    "episode_length": int(episode.episode_length),
                    "search_strategy": strategy,
                    "attempt_found": attempts
                }
                if reward != -1.0:
                    valid_models_count += 1
                
            attempts += 1
            
    finally:
        trainer.agent.load_state_dict(agent_backup)
        trainer.replay_buffer = local_buffer
    
    results_list = list(unique_models.values())
    return pd.DataFrame(results_list)
