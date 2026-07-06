from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from delphos.grammar.task import Attribute, Covariate, Task, Taste, Transformation


def configure_modelling_space(
    task: Task,
    *,
    attributes: Sequence[str | int] | None = None,
    transformations: Sequence[str | int] | None = None,
    tastes: Sequence[str | int] | None = None,
    covariates: Sequence[str | int] | None = None,
) -> Task:
    """Return a copy of a task with a smaller modelling search space."""

    return replace(
        task,
        attributes=_select_items(task.attributes, attributes),
        transformations=_select_items(task.transformations, transformations),
        tastes=_select_items(task.tastes, tastes),
        covariates=_select_items(task.covariates, covariates),
    )


def set_covariate_levels(
    task: Task,
    **levels_by_covariate: Iterable[int],
) -> Task:
    """Return a copy of a task with updated covariate levels."""

    updated_covariates = []
    for covariate in task.covariates:
        key = _normalise_name(covariate.name)
        raw_levels = None
        for requested_name, requested_levels in levels_by_covariate.items():
            if _normalise_name(requested_name) == key:
                raw_levels = requested_levels
                break
        if raw_levels is None:
            updated_covariates.append(covariate)
            continue
        updated_covariates.append(
            replace(covariate, levels=tuple(int(level) for level in raw_levels))
        )
    return replace(task, covariates=tuple(updated_covariates))


def _select_items(items, selectors):
    if selectors is None:
        return tuple(items)
    wanted = {_normalise_selector(selector) for selector in selectors}
    selected = []
    for item in items:
        item_keys = {
            _normalise_selector(item.id),
            _normalise_selector(item.name),
        }
        if item_keys & wanted:
            selected.append(item)
    missing = wanted - {
        key
        for item in items
        for key in (_normalise_selector(item.id), _normalise_selector(item.name))
    }
    if missing:
        raise ValueError(f"Unknown modelling-space selectors: {sorted(missing)}")
    return tuple(selected)


def _normalise_selector(selector: str | int | None) -> str:
    if selector is None:
        return "none"
    if isinstance(selector, int):
        return str(selector)
    return _normalise_name(selector)


def _normalise_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
