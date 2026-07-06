"""
mdp/action.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Global action grammar, masking system and state transitions for Delphos.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch
from .task import Task
from .catalogue import Catalogue
from .specification import Specification

class ActionType(Enum):
    """
    Supported action types.
    """
    TERMINATE = 0
    ADD = 1
    CHANGE = 2


@dataclass(frozen=True)
class Action:
    """
    Represents one modelling action.

    Examples
    --------
    Action(ActionType.TERMINATE)

    Action(
        ActionType.ADD, 
        attribute_id=5)

    Action(
        ActionType.CHANGE,
        attribute_id=5,
        transform_id=2,
        taste_id=2,
        covariate_id=3
    )
    """

    type: ActionType

    attribute_id: int | None = None
    transform_id: int | None = None
    taste_id: int | None = None
    covariate_id: int | None = None


class ActionSpace:
    """
    Global action catalogue.

    Responsibilities
    ----------------
    - Build global action grammar
    - Build task masks
    - Build state masks
    - Prevent repeated actions
    - Prevent cyclic trajectories
    - Apply actions to specifications
    """

    ASC_ATTRIBUTE_ID = 1
    NO_COVARIATE_ID = 0

    def __init__(
        self,
        task: Task,
        catalogue: Catalogue,
        specification: Specification,
        linear_additive: bool = True,
        device: Optional[torch.device] = None,
        
    ) -> None:

        self.task = task
        self.catalogue = catalogue
        self.specification = specification
        self.device = (torch.device("cpu") if device is None else torch.device(device))
        self.linear_additive = linear_additive

        self.attribute_ids = catalogue.attribute_ids
        self.transform_ids = catalogue.transform_ids
        self.taste_ids = catalogue.taste_ids
        self.covariate_ids = (self.NO_COVARIATE_ID, *catalogue.covariate_ids)

        self.catalogue_actions, self.action_to_idx, self.idx_to_action = self._build_catalogue_actions()
        self.num_actions = len(self.catalogue_actions)
        self.task_mask = self._build_task_mask()
        self.terminate_action_index = 0

    @property
    def n_actions(self) -> int:
        return self.num_actions

    @staticmethod
    def is_terminate(action: Action) -> bool:
        return action.type == ActionType.TERMINATE

    @staticmethod
    def is_add(action: Action) -> bool:
        return action.type == ActionType.ADD

    @staticmethod
    def is_change(action: Action) -> bool:
        return action.type == ActionType.CHANGE

    def _build_catalogue_actions(self,) -> tuple[list[Action], dict[Action, int], dict[int, Action]]:
        """
        Build global action catalogue.       
        """

        actions: list[Action] = []
        
        # add TERMINATE action
        actions.append(Action(ActionType.TERMINATE))
        
        # add ADD actions
        if not self.linear_additive:
            for attribute_id in self.attribute_ids:
                actions.append(Action(type=ActionType.ADD, attribute_id=attribute_id))

        # add CHANGE actions
        for attribute_id in self.attribute_ids:
            if attribute_id == self.ASC_ATTRIBUTE_ID:
                for covariate_id in self.covariate_ids:
                    actions.append(
                        Action(
                            type=ActionType.CHANGE,
                            attribute_id=attribute_id,
                            transform_id=1,
                            taste_id=1,
                            covariate_id=covariate_id,
                        )
                    )
                continue

            for transform_id in self.transform_ids:
                for taste_id in self.taste_ids:
                    for covariate_id in self.covariate_ids:
                        actions.append(
                            Action(
                                type=ActionType.CHANGE,
                                attribute_id=attribute_id,
                                transform_id=transform_id,
                                taste_id=taste_id,
                                covariate_id=covariate_id,
                            )
                        )

        action_to_idx = {action: idx for idx, action in enumerate(actions)}
        idx_to_action = {idx: action for idx, action in enumerate(actions)}

        return actions, action_to_idx, idx_to_action


    def _build_task_mask(self) -> torch.BoolTensor:
        """
        Build task-specific action mask.
        """

        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)

        allowed_attributes = set(self.task.attribute_ids)
        allowed_transforms = set(self.task.transform_ids)
        allowed_tastes = set(self.task.taste_ids)
        allowed_covariates = set(self.task.covariate_ids)

        for idx, action in enumerate(self.catalogue_actions):
            if self.is_terminate(action):
                mask[idx] = True
                continue

            if self.is_add(action):
                if action.attribute_id in allowed_attributes:
                    mask[idx] = True
                continue

            assert action.attribute_id is not None
            assert action.transform_id is not None
            assert action.taste_id is not None
            assert action.covariate_id is not None

            if action.attribute_id not in allowed_attributes:
                continue

            if action.transform_id not in allowed_transforms:
                continue

            if action.taste_id not in allowed_tastes:
                continue

            if action.covariate_id != self.NO_COVARIATE_ID and action.covariate_id not in allowed_covariates:
                continue

            mask[idx] = True

        return mask

    def get_action(self, action_idx: int) -> Action:
        return self.idx_to_action[action_idx]

    def get_action_idx(self, action: Action) -> int:
        return self.action_to_idx[action]
        
    def summary(self) -> dict:

        return {
            "num_actions": self.num_actions,
            "linear_additive": self.linear_additive,
            "task": self.task.name,
            "n_catalogue_attributes": len(
                self.attribute_ids
            ),
            "n_catalogue_covariates": len(
                self.covariate_ids
            ),
            "task_actions": int(
                self.task_mask.sum().item()
            ),
        }

    
    def create_initial_specification(self, include_asc: bool = True) -> torch.LongTensor:
        specification = self.specification.empty()

        if not self.linear_additive:
            return specification

        for attribute_id in self.task.attribute_ids:
            if attribute_id == self.ASC_ATTRIBUTE_ID and not include_asc:
                continue
            row = self.specification.attribute_id_to_idx[attribute_id]
            specification[row, Specification.TRANSFORM_COL] = 1
            specification[row, Specification.TASTE_COL] = 1
            specification[row, Specification.COVARIATE_COL] = self.NO_COVARIATE_ID

        return specification


    def get_state_mask(self, specification: torch.LongTensor,) -> torch.BoolTensor:
        """
        Mask actions according to current state.

        ADD:
            only inactive attributes

        CHANGE:
            only active attributes
        """

        self.specification.validate(specification)

        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device,)
        active_attributes = set()

        for row in specification:
            attribute_id = int(row[Specification.ATTRIBUTE_COL])
            transform_id = int(row[Specification.TRANSFORM_COL])
            taste_id = int(row[Specification.TASTE_COL])
            covariate_id = int(row[Specification.COVARIATE_COL])

            if transform_id == 0 and taste_id == 0 and covariate_id == 0:
                continue

            active_attributes.add(attribute_id)

        for idx, action in enumerate(self.catalogue_actions):
            if self.is_terminate(action):
                mask[idx] = True
                continue

            if self.is_add(action):
                mask[idx] = action.attribute_id not in active_attributes
                continue

            if self.is_change(action):
                mask[idx] = action.attribute_id in active_attributes

        return mask


    def get_repeated_mask(self, specification: torch.LongTensor,) -> torch.BoolTensor:
        """
        Prevent actions that reproduce the
        current active term.
        """

        self.specification.validate(specification)
        mask = torch.zeros(self.num_actions, dtype=torch.bool,device=self.device,)
        active_terms = set()

        for row in specification:
            attribute_id = int(row[Specification.ATTRIBUTE_COL])
            transform_id = int(row[Specification.TRANSFORM_COL])
            taste_id = int(row[Specification.TASTE_COL])
            covariate_id = int(row[Specification.COVARIATE_COL])

            if transform_id == 0 and taste_id == 0 and covariate_id == 0:
                continue

            active_terms.add((attribute_id, transform_id, taste_id, covariate_id))

        for idx, action in enumerate(self.catalogue_actions):
            if self.is_terminate(action):
                mask[idx] = True
                continue

            if self.is_add(action):
                mask[idx] = True
                continue

            proposed_term = (action.attribute_id, action.transform_id, action.taste_id, action.covariate_id,)
            mask[idx] = proposed_term not in active_terms

        return mask


    def get_trajectory_mask(self, specification: torch.LongTensor, visited_specifications: set[str] | None, base_mask: torch.BoolTensor) -> torch.BoolTensor:
        """
        Prevent cyclic trajectories.
        """
        mask = torch.zeros(self.num_actions, dtype=torch.bool,device=self.device,)
        valid_indices = torch.nonzero(base_mask,as_tuple=False,).flatten().tolist()

        for idx in valid_indices:
            action = self.idx_to_action[idx]
            if self.is_terminate(action):
                mask[idx] = True
                continue

            candidate_state, _ = self.apply_action(specification=specification, action_index=idx, visited_specifications=None, validate_action=False,)
            candidate_key = self.specification.specification_key(candidate_state)
            mask[idx] = candidate_key not in visited_specifications

        return mask

    def get_valid_mask(self, specification: torch.LongTensor, visited_specifications: set[str]) -> torch.BoolTensor:

        task_mask = self.task_mask
        state_mask = self.get_state_mask(specification)
        repeated_mask = self.get_repeated_mask(specification)
        base_mask = (task_mask & state_mask & repeated_mask)
        trajectory_mask = self.get_trajectory_mask(specification, visited_specifications, base_mask)
        return base_mask & trajectory_mask

    def get_valid_action_indices(self, specification: torch.LongTensor, visited_specifications: set[str]) -> list[int]:
        mask = self.get_valid_mask(specification, visited_specifications)
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()

    def get_valid_actions(self, specification: torch.LongTensor, visited_specifications: set[str]) -> list[Action]:
        indices = self.get_valid_action_indices(specification, visited_specifications)
        return [self.idx_to_action[idx] for idx in indices]

    def apply_action(self, specification: torch.LongTensor, action_index: int, visited_specifications: set[str] | None = None, validate_action: bool = True) -> tuple[torch.LongTensor, bool]:

        self.specification.validate(specification)

        if action_index < 0 or action_index >= self.num_actions:
            raise ValueError(f"Invalid action index {action_index}")
        if validate_action and visited_specifications is not None:
            valid_mask = self.get_valid_mask(specification, visited_specifications)
            if not bool(valid_mask[action_index]):
                raise ValueError(f"Action {action_index} is not valid.")

        action = self.idx_to_action[action_index]

        if self.is_terminate(action):
            return (specification.clone(), True,)

        next_specification = specification.clone()
        row = self.specification.attribute_id_to_idx[action.attribute_id]

        if self.is_add(action):
            next_specification[row, Specification.TRANSFORM_COL] = 1
            next_specification[row, Specification.TASTE_COL] = 1
            next_specification[row, Specification.COVARIATE_COL] = self.NO_COVARIATE_ID
            return (next_specification, False,)

        next_specification[row, Specification.TRANSFORM_COL] = action.transform_id
        next_specification[row, Specification.TASTE_COL] = action.taste_id
        next_specification[row, Specification.COVARIATE_COL] = action.covariate_id

        return (next_specification, False,)
