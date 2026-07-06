"""
inference/inference_few_shot_learning.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: few-shot inference learning utilities for Delphos.
"""
from __future__ import annotations

import copy
import pandas as pd
from trainer.trainer import Trainer
from inference.inference_utils import generate_episode_with_strategy

__all__ = ["run_few_shot_inference"]

VALID_MODES = {
    "policy_only",
    "encoder_only",
    "fine_tune_all",
}

VALID_SEARCH_STRATEGIES = {
    "greedy",
    "stochastic",
    "boltzmann",
    "topk",
}


def run_few_shot_inference(
    trainer: Trainer,
    task,
    adaptation_mode: str,
    search_strategy: str,
    adaptation_rounds: int = 50,
    max_attempts: int = 500,
    epsilon: float = 0.10,
    temperature: float = 1.0,
    top_k: int = 5,
    overwrite: bool = False,
) -> dict:
    """
    Runs few-shot inference allowing the agent to adapt its weights.
    Returns a dictionary containing the results.
    """
    if adaptation_mode not in VALID_MODES:
        raise ValueError(f"Invalid adaptation mode '{adaptation_mode}'. Expected one of {VALID_MODES}")

    if search_strategy not in VALID_SEARCH_STRATEGIES:
        raise ValueError(f"Invalid search strategy '{search_strategy}'. Expected one of {VALID_SEARCH_STRATEGIES}")

    runtime = trainer.add_task(task=task, overwrite=overwrite)
    
    agent_backup = copy.deepcopy(trainer.agent.state_dict())
    local_buffer = copy.deepcopy(trainer.replay_buffer)

    try:
        # 1. Initial Evaluation (Greedy, zero-shot state)
        trainer.agent.inference_mode()
        initial_episode = generate_episode_with_strategy(trainer, runtime, "greedy", epsilon, temperature, top_k)
        initial_reward = float(initial_episode.reward)
        initial_spec = initial_episode.specification_key
        
        # 2. Adaptation Phase
        if adaptation_rounds > 0:
            trainer.agent.set_mode(adaptation_mode)
            valid_rounds = 0
            attempts = 0
            while valid_rounds < adaptation_rounds and attempts < max_attempts:
                episode = generate_episode_with_strategy(trainer, runtime, search_strategy, epsilon, temperature, top_k)
                trainer.store_episode(episode)
                trainer.update_from_replay()
                
                if float(episode.reward) != -1.0:
                    valid_rounds += 1
                attempts += 1

        # 3. Final Evaluation (Greedy, post-adaptation)
        trainer.agent.inference_mode()
        final_episode = generate_episode_with_strategy(trainer, runtime, "greedy", epsilon, temperature, top_k)
        final_reward = float(final_episode.reward)
        final_spec = final_episode.specification_key

        result = {
            "task_id": int(runtime.task.id),
            "task_name": str(runtime.task.name),
            "experiment": f"{adaptation_mode}_{search_strategy}",
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "improvement": final_reward - initial_reward,
            "initial_specification": initial_spec,
            "final_specification": final_spec,
            "episode_length": int(final_episode.episode_length),
            "adaptation_rounds": int(adaptation_rounds),
            "adaptation_mode": adaptation_mode,
            "search_strategy": search_strategy,
            "epsilon": epsilon,
            "temperature": temperature,
            "top_k": top_k,
        }
       
    finally:
        trainer.agent.load_state_dict(agent_backup)
        trainer.replay_buffer = local_buffer
    
    return result
