"""
mdp/task_context.py

: author: Gabriel Nova
: date: Jun 2026
: version: 2.0.0
: purpose: Lightweight task-specific context used by training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from mdp.task import Task
from .action import ActionSpace
from .state import SpecificationManager


@dataclass(frozen=True)
class TaskContext:
    task: Task
    action_space: ActionSpace
    specification_manager: SpecificationManager

    @property
    def task_id(self) -> int:
        return self.task.id

    @property
    def name(self) -> str:
        return self.task.name


def build_task_context(task: Task) -> TaskContext:

    specification_manager = SpecificationManager(task=task)

    action_space = ActionSpace(task=task)

    return TaskContext(
        task=task,
        action_space=action_space,
        specification_manager=specification_manager,
    )


def build_task_contexts(tasks: Sequence[Task]) -> Dict[int, TaskContext]:

    contexts: Dict[int, TaskContext] = {}

    for task in tasks:
        contexts[task.id] = build_task_context(task)

    return contexts