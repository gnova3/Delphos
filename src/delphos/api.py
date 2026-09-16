from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from delphos.agent.checkpoint import load_agent_from_checkpoint
from delphos.data.builder import create_dataset
from delphos.data.registry import (
    default_checkpoint_path,
    list_datasets,
    load_dataset,
    load_global_catalogue,
)
from delphos.data.validation import validate_dataset_config
from delphos.grammar.runtime import build_runtime
from delphos.grammar.space import configure_modelling_space, set_covariate_levels
from delphos.grammar.task import Task
from delphos.inference.proposal import ProposalSet, propose_models as _propose_models


@dataclass
class DelphosModel:
    """Loaded Delphos checkpoint plus the global modelling catalogue."""

    agent: Any
    catalogue: Any
    checkpoint_path: Path

    def propose(
        self,
        dataset: str | int | Path | Task,
        *,
        n_models: int = 25,
        estimate: bool = False,
        estimate_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        device: str | None = None,
        **advanced_kwargs,
    ) -> ProposalSet:
        task = _coerce_task(dataset)
        _validate_task_against_catalogue(task, self.catalogue)
        
        linear_additive = advanced_kwargs.get("linear_additive", True)
        
        runtime = build_runtime(
            task=task,
            catalogue=self.catalogue,
            linear_additive=linear_additive,
            device=str(device or self.agent.device),
        )
        return _propose_models(
            agent=self.agent,
            runtime=runtime,
            n_models=n_models,
            max_attempts=advanced_kwargs.get("max_attempts", 250),
            strategy=advanced_kwargs.get("strategy", "topk"),
            epsilon=advanced_kwargs.get("epsilon", 0.10),
            temperature=advanced_kwargs.get("temperature", 1.0),
            top_k=advanced_kwargs.get("top_k", 5),
            horizon_kappa=advanced_kwargs.get("horizon_kappa", 2.0),
            estimate=estimate,
            estimate_kwargs=estimate_kwargs,
            seed=seed,
        )


def load_agent(
    checkpoint: str | Path | None = None,
    *,
    device: str = "cpu",
) -> DelphosModel:
    checkpoint_path = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
    agent = load_agent_from_checkpoint(checkpoint_path, device=device)
    catalogue = load_global_catalogue()
    return DelphosModel(agent=agent, catalogue=catalogue, checkpoint_path=checkpoint_path)


def propose_models(
    dataset: str | int | Path | Task,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    **kwargs,
) -> ProposalSet:
    model = load_agent(checkpoint=checkpoint, device=device)
    return model.propose(dataset, **kwargs)


def _coerce_task(dataset: str | int | Path | Task) -> Task:
    if isinstance(dataset, Task):
        return dataset
    if isinstance(dataset, Path):
        return Task.from_folder(id=_task_id_from_path(dataset), folder=dataset)
    if isinstance(dataset, int):
        return load_dataset(dataset)

    dataset_str = str(dataset)
    dataset_path = Path(dataset_str)
    if dataset_path.exists():
        return Task.from_folder(id=_task_id_from_path(dataset_path), folder=dataset_path)
    return load_dataset(dataset_str)


def _task_id_from_path(path: Path) -> int:
    name = path.name
    if name.startswith("dataset_"):
        suffix = name.removeprefix("dataset_")
        if suffix.isdigit():
            return int(suffix)
    return abs(hash(str(path.resolve()))) % 1_000_000


def _validate_task_against_catalogue(task: Task, catalogue) -> None:
    missing_attributes = sorted(set(task.attribute_ids) - set(catalogue.attribute_ids))
    missing_covariates = sorted(set(task.covariate_ids) - set(catalogue.covariate_ids))
    missing_transforms = sorted(set(task.transform_ids) - set(catalogue.transform_ids))
    missing_tastes = sorted(set(task.taste_ids) - set(catalogue.taste_ids))

    problems = []
    if missing_attributes:
        problems.append(f"attributes={missing_attributes}")
    if missing_covariates:
        problems.append(f"covariates={missing_covariates}")
    if missing_transforms:
        problems.append(f"transforms={missing_transforms}")
    if missing_tastes:
        problems.append(f"tastes={missing_tastes}")
    if problems:
        raise ValueError(
            "Dataset uses ids outside the trained Delphos catalogue: "
            + ", ".join(problems)
        )
