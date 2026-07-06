"""
mdp/runtime.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Runtime container used by RL algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Sequence, Callable, Optional
from .task import Task
from .catalogue import Catalogue
from .specification import Specification
from .actions import ActionSpace

@dataclass(frozen=True)
class Runtime:
    """
    Runtime environment associated with one task.
    """

    task: Task
    catalogue: Catalogue
    specification: Specification
    action_space: ActionSpace
    context_vector: torch.FloatTensor


def build_runtime(task: Task, catalogue: Catalogue, linear_additive: bool = False, device: str = "cpu") -> Runtime:

    specification = Specification(catalogue, device)
    action_space = ActionSpace(task, catalogue, specification, linear_additive, device)

    _attribute_mask = torch.as_tensor(catalogue.attribute_mask(task), dtype=torch.float32, device=device,)
    _covariate_mask = torch.as_tensor(catalogue.covariate_mask(task), dtype=torch.float32, device=device,)

    context_vector = torch.cat([_attribute_mask, _covariate_mask], dim=0)

    return Runtime(task, catalogue, specification, action_space, context_vector)


def build_runtimes(tasks: Sequence[Task], linear_additive: bool = False, device: str = "cpu") -> tuple[dict[int, Runtime], Catalogue]:

    catalogue = Catalogue.from_tasks(tasks)

    runtimes = {}
    for task in tasks:
        runtimes[task.id] = build_runtime(task, catalogue, linear_additive, device)

    return runtimes, catalogue

def evaluate(runtime: Runtime, specification: torch.LongTensor, return_specification_key: bool = False, suppress_exceptions: bool = True, logger: Optional[object] = None,debug_apollo = False):

    try:

        runtime.specification.validate(specification)
        backend = runtime.specification.to_backend(specification)    
        from delphos.env.apollo.generator import ApolloGenerator
        from delphos.env.environment import evaluate_specification
        from delphos.env.reward import reward_function

        generator = ApolloGenerator(runtime.task)    
        apollo_specification = generator.build_apollo_specification(backend)
        
        debug_path = runtime.task.dataset_path.parent / "outputs" / "debug" / backend["key"]
        modelling_outcome = evaluate_specification(
            task=runtime.task,
            apollo_specification=apollo_specification,
            debug_apollo=debug_apollo,
            debug_path=debug_path,
        )

        reward = reward_function(task=runtime.task, modelling_outcome=modelling_outcome)

        if return_specification_key:
            return backend["key"], reward
        return reward

    except Exception as exc:
        print("\nRUNTIME EVALUATION ERROR")
        print(type(exc))
        print(exc)
        if logger is not None:
            logger.exception("evaluate_specification failed for task=%s", runtime.task.name)
        if suppress_exceptions:
            if return_specification_key:
                return "", -1.0
            return -1.0
        raise exc
