"""
trainer/episode.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Episode generation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from agent.delphos import Delphos
from agent.replay_buffer import Transition
from mdp.runtime import Runtime, evaluate


@dataclass
class EpisodeResult:
    transitions: list[Transition]
    reward: float
    specification: torch.LongTensor
    specification_key: str
    episode_length: int
    horizon: int
    terminated: bool
    forced_stop: bool
    task_id: int
    task_name: str
    action_indices: list[int]
    visited_specifications: tuple[str, ...]
    valid_actions_per_step: list[int]
    z_states: list[np.ndarray]
    unique_specifications: int
    unique_actions: int
    runtime_seconds: float


def compute_horizon(runtime: Runtime, horizon_kappa: float = 2.0,) -> int:
    n_attributes = len(runtime.task.attribute_ids)
    horizon = int(np.ceil(horizon_kappa * n_attributes))
    return max(1, horizon)


def distribute_terminal_reward(reward: float, trajectory_length: int, gamma: float) -> list[float]:
    if trajectory_length <= 0:
        return [float(reward)]
    return [float(gamma ** (trajectory_length - i - 1) * reward) for i in range(trajectory_length)]


@torch.no_grad()
def run_episode(
    agent: Delphos,
    runtime: Runtime,
    epsilon: float,
    discount_factor: float,
    horizon_kappa: float = 2.0,
    boltzmann: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> EpisodeResult:

    action_space = runtime.action_space
    specification_space = runtime.specification
    specification = action_space.create_initial_specification()
    specification_key = specification_space.specification_key(specification)
    visited_specifications = {specification_key}
    
    transitions_raw = []
    action_indices = []
    valid_actions_per_step = []
    z_states = []
    terminated = False

    H_tau = compute_horizon(runtime=runtime, horizon_kappa=horizon_kappa)
    start_time = time.time()

    while True:
        if terminated:
            break
        if len(transitions_raw) >= H_tau:
            break
        
        z_state = agent.encode_specification(specification=specification,runtime=runtime).detach().cpu()
        z_states.append(z_state.numpy())
        current_valid_indices = action_space.get_valid_action_indices(specification=specification,visited_specifications=visited_specifications)
        valid_actions_per_step.append(len(current_valid_indices))

        if top_k is None:
            action_index, _ = agent.select_action(
                specification=specification,
                runtime=runtime,
                visited_specifications=visited_specifications,
                epsilon=epsilon,
                boltzmann=boltzmann,
                temperature=temperature,
            )
        else:
            action_index, _ = agent.topk_action(
                specification=specification,
                runtime=runtime,
                visited_specifications=visited_specifications,
                k=top_k,
                temperature=temperature,
            )

        next_specification, terminated = action_space.apply_action(
            specification=specification,
            action_index=action_index,
            visited_specifications=visited_specifications,
        )

        next_key = specification_space.specification_key(next_specification)
        next_visited = set(visited_specifications)
        next_visited.add(next_key)

        if terminated:
            next_valid_indices = []
        else:
            next_valid_indices = action_space.get_valid_action_indices(specification=next_specification,visited_specifications=next_visited)

        transitions_raw.append({
            "state": specification.clone(),
            "z_state": z_state.clone(),
            "action": int(action_index),
            "next_state": next_specification.clone(),
            "done": bool(terminated),
            "next_valid_indices": next_valid_indices,
        })

        action_indices.append(int(action_index))
        specification = next_specification
        visited_specifications = next_visited

    final_specification = specification
    episode_length = len(transitions_raw)
    forced_stop = (episode_length >= H_tau) and (not terminated)

    specification_key, reward = evaluate(runtime=runtime,specification=final_specification,return_specification_key=True, 
                                         suppress_exceptions= False, logger= None,debug_apollo = False)
    transition_rewards = distribute_terminal_reward(reward=float(reward),trajectory_length=episode_length,gamma=discount_factor)

    transitions: list[Transition] = []
    for step_idx, transition in enumerate(transitions_raw):
        next_state = transition["next_state"]
        z_next_state = agent.encode_specification(specification=next_state,runtime=runtime).detach().cpu()
        transitions.append(
            Transition(
                task_id=runtime.task.id,
                state=transition["state"],
                z_state=transition["z_state"],
                action_index=transition["action"],
                reward=float(transition_rewards[step_idx]),
                next_state=next_state,
                z_next_state=z_next_state,
                done=bool(step_idx == episode_length - 1),
                next_valid_indices=transition["next_valid_indices"],
            )
        )

    return EpisodeResult(
        transitions=transitions,
        reward=float(reward),
        specification=final_specification,
        specification_key=str(specification_key),
        episode_length=int(episode_length),
        horizon=int(H_tau),
        terminated=bool(terminated),
        forced_stop=bool(forced_stop),
        task_id=int(runtime.task.id),
        task_name=str(runtime.task.name),
        action_indices=action_indices,
        visited_specifications=tuple(visited_specifications),
        valid_actions_per_step=valid_actions_per_step,
        z_states=z_states,
        unique_specifications=int(len(visited_specifications))  ,
        unique_actions=int(len(set(action_indices))),
        runtime_seconds=float(time.time() - start_time),
    )