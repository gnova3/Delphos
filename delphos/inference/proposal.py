from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from delphos.env.apollo.generator import ApolloGenerator


@dataclass
class Proposal:
    task_id: int
    task_name: str
    specification_key: str
    terms: list[Any]
    action_indices: list[int]
    episode_length: int
    search_strategy: str
    attempt_found: int
    estimated: bool = False
    reward: float | None = None
    outcome: Any | None = None
    apollo_specification: Any | None = None


@dataclass
class ProposalSet:
    proposals: list[Proposal] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.proposals)

    def __iter__(self):
        return iter(self.proposals)

    def keys(self) -> list[str]:
        return [proposal.specification_key for proposal in self.proposals]

    def to_records(self) -> list[dict[str, Any]]:
        records = []
        for proposal in self.proposals:
            row = {
                "task_id": proposal.task_id,
                "task_name": proposal.task_name,
                "specification_key": proposal.specification_key,
                "episode_length": proposal.episode_length,
                "search_strategy": proposal.search_strategy,
                "attempt_found": proposal.attempt_found,
                "estimated": proposal.estimated,
                "reward": proposal.reward,
                "n_terms": len(proposal.terms),
                "action_indices": proposal.action_indices,
            }
            if proposal.outcome is not None and not proposal.outcome.empty:
                for column, value in proposal.outcome.iloc[0].items():
                    row[column] = value
            records.append(row)
        return records

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def estimate(self, task, **estimate_kwargs) -> "ProposalSet":
        """Estimate all proposals through the Delphos environment."""

        from delphos.data.registry import load_dataset
        from delphos.env.environment import evaluate_specification
        from delphos.env.reward import reward_function

        if isinstance(task, (str, int)):
            task = load_dataset(task)

        for proposal in self.proposals:
            outcome = evaluate_specification(
                task=task,
                apollo_specification=proposal.apollo_specification,
                **estimate_kwargs,
            )
            proposal.outcome = outcome
            proposal.reward = reward_function(task, outcome)
            proposal.estimated = True
        return self


def propose_models(
    *,
    agent,
    runtime,
    n_models: int = 25,
    max_attempts: int = 250,
    strategy: str = "topk",
    epsilon: float = 0.10,
    temperature: float = 1.0,
    top_k: int = 5,
    horizon_kappa: float = 2.0,
    estimate: bool = False,
    estimate_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> ProposalSet:
    strategy = strategy.lower()
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    unique: dict[str, Proposal] = {}
    attempts = 0
    agent.inference_mode()

    while len(unique) < int(n_models) and attempts < int(max_attempts):
        specification, action_indices = _generate_specification(
            agent=agent,
            runtime=runtime,
            strategy=strategy,
            epsilon=epsilon,
            temperature=temperature,
            top_k=top_k,
            horizon_kappa=horizon_kappa,
        )
        specification_key = runtime.specification.specification_key(specification)
        if specification_key not in unique:
            unique[specification_key] = _build_proposal(
                runtime=runtime,
                specification=specification,
                action_indices=action_indices,
                strategy=strategy,
                attempt=attempts,
                estimate=estimate,
                estimate_kwargs=estimate_kwargs or {},
            )
        attempts += 1

    return ProposalSet(list(unique.values()))


def _generate_specification(
    *,
    agent,
    runtime,
    strategy: str,
    epsilon: float,
    temperature: float,
    top_k: int,
    horizon_kappa: float,
):
    action_space = runtime.action_space
    specification_space = runtime.specification
    specification = action_space.create_initial_specification()
    specification_key = specification_space.specification_key(specification)
    visited_specifications = {specification_key}
    action_indices: list[int] = []
    horizon = _compute_horizon(runtime=runtime, horizon_kappa=horizon_kappa)

    for _ in range(horizon):
        action_index = _select_action(
            agent=agent,
            runtime=runtime,
            specification=specification,
            visited_specifications=visited_specifications,
            strategy=strategy,
            epsilon=epsilon,
            temperature=temperature,
            top_k=top_k,
        )
        next_specification, terminated = action_space.apply_action(
            specification=specification,
            action_index=action_index,
            visited_specifications=visited_specifications,
        )
        action_indices.append(int(action_index))
        specification = next_specification
        specification_key = specification_space.specification_key(specification)
        visited_specifications.add(specification_key)
        if terminated:
            break

    return specification, action_indices


def _select_action(
    *,
    agent,
    runtime,
    specification,
    visited_specifications: set[str],
    strategy: str,
    epsilon: float,
    temperature: float,
    top_k: int,
) -> int:
    if strategy == "greedy":
        action_index, _ = agent.select_action(
            specification=specification,
            runtime=runtime,
            visited_specifications=visited_specifications,
            epsilon=0.0,
            boltzmann=False,
        )
        return action_index
    if strategy == "stochastic":
        action_index, _ = agent.select_action(
            specification=specification,
            runtime=runtime,
            visited_specifications=visited_specifications,
            epsilon=epsilon,
            boltzmann=False,
        )
        return action_index
    if strategy == "boltzmann":
        action_index, _ = agent.select_action(
            specification=specification,
            runtime=runtime,
            visited_specifications=visited_specifications,
            epsilon=0.0,
            boltzmann=True,
            temperature=temperature,
        )
        return action_index
    if strategy == "topk":
        action_index, _ = agent.topk_action(
            specification=specification,
            runtime=runtime,
            visited_specifications=visited_specifications,
            k=top_k,
            temperature=temperature,
        )
        return action_index
    raise ValueError("Unknown strategy. Expected greedy, stochastic, boltzmann, or topk.")


def _build_proposal(
    *,
    runtime,
    specification,
    action_indices: list[int],
    strategy: str,
    attempt: int,
    estimate: bool,
    estimate_kwargs: dict[str, Any],
) -> Proposal:
    backend = runtime.specification.to_backend(specification)
    generator = ApolloGenerator(runtime.task)
    apollo_specification = generator.build_apollo_specification(backend)
    terms = runtime.specification.to_terms(specification)

    outcome = None
    reward = None
    if estimate:
        from delphos.env.environment import evaluate_specification
        from delphos.env.reward import reward_function

        outcome = evaluate_specification(
            task=runtime.task,
            apollo_specification=apollo_specification,
            **estimate_kwargs,
        )
        reward = reward_function(runtime.task, outcome)

    return Proposal(
        task_id=int(runtime.task.id),
        task_name=str(runtime.task.name),
        specification_key=str(backend["key"]),
        terms=terms,
        action_indices=action_indices,
        episode_length=len(action_indices),
        search_strategy=strategy,
        attempt_found=int(attempt),
        estimated=bool(estimate),
        reward=reward,
        outcome=outcome,
        apollo_specification=apollo_specification,
    )


def _compute_horizon(runtime, horizon_kappa: float = 2.0) -> int:
    n_attributes = len(runtime.task.attribute_ids)
    horizon = int(np.ceil(float(horizon_kappa) * n_attributes))
    return max(1, horizon)
