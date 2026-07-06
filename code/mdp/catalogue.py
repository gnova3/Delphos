"""
mdp/catalogue.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Builds the global modelling grammar across tasks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np
from .task import Task


@dataclass(frozen=True)
class Catalogue:
    """
    Aggregates modelling terms across tasks.
    Creates a global identifier for each term.

    Notes
    -----
    IDs are assumed to be user-defined and globally consistent.

    """

    tasks: tuple[Task, ...]

    attribute_ids: tuple[int, ...]
    covariate_ids: tuple[int, ...]
    transform_ids: tuple[int, ...]
    taste_ids: tuple[int, ...]

    attribute_id_to_idx: dict[int, int]
    covariate_id_to_idx: dict[int, int]
    transform_id_to_idx: dict[int, int]
    taste_id_to_idx: dict[int, int]

    task_id_to_idx: dict[int, int]

    # ======================================================
    # Constructors
    # ======================================================
    @classmethod
    def from_tasks(cls, tasks: Sequence[Task]) -> Catalogue:
        """
        Build catalogue from a collection of tasks.
        """

        tasks = tuple(tasks)

        attribute_ids = tuple(sorted({attribute.id for task in tasks for attribute in task.attributes}))
        covariate_ids = tuple(sorted({covariate.id for task in tasks for covariate in task.covariates if covariate.id is not None}))
        transform_ids = tuple(sorted({transform.id for task in tasks for transform in task.transformations}))
        taste_ids = tuple(sorted({taste.id for task in tasks for taste in task.tastes}))

        catalogue = cls(
            tasks=tasks,

            attribute_ids=attribute_ids,
            covariate_ids=covariate_ids,
            transform_ids=transform_ids,
            taste_ids=taste_ids,

            attribute_id_to_idx={id_: idx for idx, id_ in enumerate(attribute_ids)},
            covariate_id_to_idx={id_: idx for idx, id_ in enumerate(covariate_ids)},
            transform_id_to_idx={id_: idx for idx, id_ in enumerate(transform_ids)},
            taste_id_to_idx={id_: idx for idx, id_ in enumerate(taste_ids)},
            
            task_id_to_idx={task.id: idx for idx, task in enumerate(tasks)}
        )
        catalogue.validate()
        return catalogue

    # ======================================================
    # Validation
    # ======================================================
    def validate(self) -> None:
        """
        Validate catalogue consistency.
        """
        if len(self.tasks) == 0:
            raise ValueError("Catalogue requires at least one task.")

        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task ids detected.")

    # ======================================================
    # Global IDs
    # ======================================================
    @property
    def global_attribute_ids(self) -> tuple[int, ...]:
        return self.attribute_ids

    @property
    def global_covariate_ids(self) -> tuple[int, ...]:
        return self.covariate_ids

    @property
    def global_transform_ids(self) -> tuple[int, ...]:
        return self.transform_ids

    @property
    def global_taste_ids(self) -> tuple[int, ...]:
        return self.taste_ids
    
    # ======================================================
    # Dimensionalities
    # ======================================================
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_attributes(self) -> int:
        return len(self.attribute_ids)

    @property
    def n_covariates(self) -> int:
        return len(self.covariate_ids)

    @property
    def n_transformations(self) -> int:
        return len(self.transform_ids)

    @property
    def n_tastes(self) -> int:
        return len(self.taste_ids)

    # ======================================================
    # Task lookup
    # ======================================================
    def get_task(self, task_id: int) -> Task:
        return self.tasks[self.task_id_to_idx[task_id]]

    # ======================================================
    # Mask helper
    # ======================================================

    def _mask(self, task_ids: tuple[int, ...], global_ids: tuple[int, ...], id_to_idx: dict[int, int]) -> np.ndarray:
        mask = np.zeros(len(global_ids), dtype=bool)
        for id_ in task_ids:
            mask[id_to_idx[id_]] = True
        return mask

    # ======================================================
    # Task masks
    # ======================================================
    def attribute_mask(self, task: Task) -> np.ndarray:
        return self._mask(task.attribute_ids, self.attribute_ids, self.attribute_id_to_idx)

    def covariate_mask(self, task: Task) -> np.ndarray:
        return self._mask(task.covariate_ids, self.covariate_ids, self.covariate_id_to_idx)

    def transform_mask(self, task: Task) -> np.ndarray:
        return self._mask(task.transform_ids, self.transform_ids, self.transform_id_to_idx)

    def taste_mask(self, task: Task) -> np.ndarray:
        return self._mask(task.taste_ids, self.taste_ids, self.taste_id_to_idx)   

    # ======================================================
    # Representation
    # ======================================================

    def summary(self) -> dict[str, Any]:
        return (
            f"Catalogue("
            f"tasks={self.n_tasks}, "
            f"attributes={self.n_attributes}, "
            f"covariates={self.n_covariates}, "
            f"transformations={self.n_transformations}, "
            f"tastes={self.n_tastes}"
            f")"
        )