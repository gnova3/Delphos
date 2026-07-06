# =============================================================================
#                       Replay buffers
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import inspect
import random
import numpy as np

__all__ = ["Transition", "UniformReplayBuffer", "PrioritizedReplayBuffer", "RankBasedReplayBuffer", "make_replay_buffer",]

# -----------------------------------------------------------------------------
# Transition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Transition:
    """Single experience tuple stored in replay memory.

        A transition represents one interaction step between the agent and the environment.

        The replay buffer stores transitions from all tasks in a shared memory structure, 
        enabling both multitask learning and balanced sampling.

        Attributes:
            task_id: Identifier of the task that generated the transition.
            state: State representation before executing the action.
            z_state: Optional encoded latent representation of the state.
            action_index: Index of the selected action in the global action catalogue.
            reward: Immediate reward obtained after executing the action.
            next_state: Environment state after executing the action.
            z_next_state: Optional encoded latent representation of the next state.
            done: Whether the transition terminates the episode.
            next_valid_indices: Valid action indices available in the next state for masking.
        """
    task_id: int
    state: Any
    z_state: Any
    action_index: int
    reward: float
    next_state: Any
    z_next_state: Any
    done: bool
    next_valid_indices: list[int] = field(default_factory=list)

def make_replay_buffer(capacity: int, kind: str = "uniform", **kwargs,):
    """Create a replay buffer based on a user-config.

    Supported replay buffer types:
    - uniform      : Uniform random sampling.
    - prioritized  : Proportional Prioritized Experience Replay (PER).
    - rank         : Rank-based Prioritized Experience Replay.

    Args:
        capacity: Maximum number of transitions stored [int].
        kind: Replay buffer type identifier [str].
        **kwargs: Additional arguments forwarded to the selected buffer.

    Returns: Replay buffer instance.

    Raises:
        ValueError: If the requested replay buffer type is unknown.
    """
    k = (kind or "uniform").lower()

    if k in {"uniform", "u"}:
        return UniformReplayBuffer(capacity=capacity, **_filter_kwargs(kwargs, UniformReplayBuffer),)
    
    if k in {"per", "proportional", "prioritized", "p"}:
        return PrioritizedReplayBuffer(capacity=capacity, **_filter_kwargs(kwargs, PrioritizedReplayBuffer),)
    
    if k in {"rank", "ranked", "r"}:
        return RankBasedReplayBuffer(capacity=capacity, **_filter_kwargs(kwargs, RankBasedReplayBuffer),)
    
    raise ValueError(f"Unknown replay buffer kind: {kind!r}")


def _filter_kwargs(kwargs: dict, cls):
    """Filter unsupported keyword arguments

    Removes arguments that are not accepted by the target class
    constructor. This allows a common configuration dictionary to be
    reused across different replay buffer implementations.
    """
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class _TaskBalancedMixin:
    # TODO: Improve this
    """Shared utilities for multitask replay buffers.

    Provides structures required for balanced sampling
    across tasks when a single replay buffer stores transitions from
    multiple datasets.

    The mixin maintains:
    - task → buffer indices
    - buffer index → task
    - balanced batch allocation

    This enables Delphos to avoid over-sampling transitions from large
    datasets while under-sampling smaller tasks.
    """

    def _init_task_tracking(self) -> None:
        """Initialize the structures for tracking which buffer indices belong to which task ids."""
        self.index_to_task = np.full(self.capacity, -1, dtype=np.int32)
        self.task_to_indices: dict[int, set[int]] = {}

    def _register_index(self, buffer_index: int, task_id: int) -> None:
        """Register a buffer index for a given task id, updating the tracking structures."""
        old_task_id = int(self.index_to_task[buffer_index])
        if old_task_id != -1:
            old_set = self.task_to_indices.get(old_task_id)
            if old_set is not None:
                old_set.discard(buffer_index)
                if len(old_set) == 0:
                    self.task_to_indices.pop(old_task_id, None)

        self.index_to_task[buffer_index] = int(task_id)
        self.task_to_indices.setdefault(int(task_id), set()).add(buffer_index)

    def _active_task_ids(self) -> list[int]:
        """Return a sorted list of task ids that currently have at least one transition in the buffer."""
        return sorted(task_id for task_id, idxs in self.task_to_indices.items() if len(idxs) > 0)

    def _resolve_task_ids(self, task_ids: Optional[list[int]] = None) -> list[int]:
        """Resolve the provided task ids for balanced sampling, defaulting to all active task ids if None."""
        if task_ids is None:
            resolved = self._active_task_ids()
        else:
            resolved = [int(task_id) for task_id in task_ids if int(task_id) in self.task_to_indices and len(self.task_to_indices[int(task_id)]) > 0]

        if not resolved:
            raise ValueError("No active task ids available for balanced sampling.")
        return resolved

    def _balanced_counts(self, batch_size: int, task_ids: list[int]) -> dict[int, int]:
        """Given a batch size and a list of task ids, compute how many samples to draw from each task for balanced sampling."""
        n_tasks = len(task_ids)
        base = batch_size // n_tasks
        rem = batch_size % n_tasks
        counts = {task_id: base for task_id in task_ids}
        for task_id in task_ids[:rem]:
            counts[task_id] += 1
        return counts

    def transitions_per_task(self) -> dict[int, int]:
        return {task_id: len(indices) for task_id, indices in self.task_to_indices.items()}

    def summary(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": len(self),
            "num_tasks": len(self._active_task_ids()),
            "transitions_per_task": self.transitions_per_task(),
        }


# =============================================================================
#   Uniform Replay Buffer
# =============================================================================
class UniformReplayBuffer(_TaskBalancedMixin):
    """Uniform replay memory.

    Stores transitions in a buffer and samples experiences uniformly.

    This is the simplest replay strategy and serves as a baseline.

    Supports both:
        - standard random sampling
        - balanced multitask sampling
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: list[Optional[Transition]] = [None] * self.capacity
        self.write = 0
        self.size = 0
        self._init_task_tracking()

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition, priority: Optional[float] = None) -> None:
        self.data[self.write] = transition
        self._register_index(self.write, transition.task_id)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int,) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        if len(self) < batch_size:
            return None
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.data[i] for i in idxs]
        is_w = np.ones(batch_size, dtype=np.float32)
        return batch, idxs.astype(np.int32), is_w

    def sample_balanced(self, batch_size: int, task_ids: Optional[list[int]] = None,) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        active_task_ids = self._resolve_task_ids(task_ids)
        if len(self) < batch_size:
            return None

        counts = self._balanced_counts(batch_size, active_task_ids)
        sampled_indices: list[int] = []

        for task_id, n_draw in counts.items():
            candidates = np.array(sorted(self.task_to_indices[task_id]), dtype=np.int32)
            if candidates.size == 0:
                return None

            replace = candidates.size < n_draw
            chosen = np.random.choice(candidates, size=n_draw, replace=replace)
            sampled_indices.extend(int(i) for i in chosen)

        random.shuffle(sampled_indices)
        indices = np.asarray(sampled_indices, dtype=np.int32)
        batch = [self.data[i] for i in indices]
        is_w = np.ones(len(indices), dtype=np.float32)
        return batch, indices, is_w

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        return


# =============================================================================
#   Proportional PER (shared buffer + task tracking)
# =============================================================================
class SumTree:
    """Binary tree data structure for proportional PER.

    Stores transition priorities in leaf nodes and cumulative priority
    sums in internal nodes.

    Enables O(log N):
    - insertion
    - priority updates
    - priority-proportional sampling
    """
    def __init__(self, capacity: int) -> None:
        assert capacity > 1, "capacity must be > 1"
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        self.data: list[Optional[Transition]] = [None] * self.capacity
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Transition) -> tuple[int, int]:
        data_idx = self.write
        leaf_idx = data_idx + self.capacity - 1
        self.data[data_idx] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return leaf_idx, data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        assert priority >= 0.0
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, mass: float) -> tuple[int, int, float, Transition]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = idx
                break
            if mass <= self.tree[left] or self.tree[right] == 0.0:
                idx = left
            else:
                mass -= self.tree[left]
                idx = right

        data_idx = leaf_idx - (self.capacity - 1)
        data = self.data[data_idx]
        assert data is not None
        return leaf_idx, data_idx, float(self.tree[leaf_idx]), data


class PrioritizedReplayBuffer(_TaskBalancedMixin):
    """Proportional Prioritized Experience Replay (PER).

    Samples transitions with probability proportional to:
        p_i = (|TD error| + ε)^α
    and corrects the resulting sampling bias using importance
    sampling weights.

    Supports:
    - standard PER sampling
    - balanced multitask PER sampling
    - priority updates from TD errors
    """
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta1: float = 1.0,
        beta_anneal_steps: int = 200_000,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = int(capacity)
        self.tree = SumTree(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.beta1 = float(beta1)
        self.beta_inc = (self.beta1 - self.beta) / max(1, int(beta_anneal_steps))
        self.eps = float(eps)
        self._max_priority = 1.0
        self._init_task_tracking()

    def __len__(self) -> int:
        return self.tree.n_entries

    def _priority(self, td_or_priority: Optional[float]) -> float:
        base = self._max_priority if td_or_priority is None else abs(float(td_or_priority)) + self.eps
        return base ** self.alpha

    def add(self, transition: Transition, priority: Optional[float] = None) -> None:
        p = self._priority(priority)
        _, data_idx = self.tree.add(p, transition)
        self._register_index(data_idx, transition.task_id)
        self._max_priority = max(self._max_priority, p)

    def add_episode(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(
        self,
        batch_size: int,
    ) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        if len(self) < batch_size:
            return None

        batch: list[Transition] = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)

        segment = self.tree.total / batch_size
        segment = segment if segment > 0.0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a, b)
            leaf_idx, _, p, data = self.tree.get(mass)
            batch.append(data)
            indices[i] = leaf_idx
            priorities[i] = max(p, self.eps)

        total = max(self.tree.total, self.eps)
        probs = priorities / total
        n = float(len(self))
        is_w = (n * probs) ** (-self.beta)
        is_w /= is_w.max()
        self.beta = min(self.beta1, self.beta + self.beta_inc)
        return batch, indices, is_w.astype(np.float32)

    def sample_balanced(
        self,
        batch_size: int,
        task_ids: Optional[list[int]] = None,
    ) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        active_task_ids = self._resolve_task_ids(task_ids)
        if len(self) < batch_size:
            return None

        counts = self._balanced_counts(batch_size, active_task_ids)

        batch: list[Transition] = []
        leaf_indices: list[int] = []
        priorities: list[float] = []

        valid_leaf_offset = self.capacity - 1

        for task_id, n_draw in counts.items():
            candidate_data_indices = np.array(sorted(self.task_to_indices[task_id]), dtype=np.int32)
            if candidate_data_indices.size == 0:
                return None

            candidate_leaf_indices = candidate_data_indices + valid_leaf_offset
            candidate_priorities = self.tree.tree[candidate_leaf_indices].astype(np.float64)
            candidate_priorities = np.maximum(candidate_priorities, self.eps)

            probs = candidate_priorities / candidate_priorities.sum()
            replace = candidate_leaf_indices.size < n_draw
            chosen_pos = np.random.choice(
                np.arange(candidate_leaf_indices.size),
                size=n_draw,
                replace=replace,
                p=probs,
            )

            for pos in chosen_pos:
                leaf_idx = int(candidate_leaf_indices[pos])
                data_idx = leaf_idx - valid_leaf_offset
                transition = self.tree.data[data_idx]
                assert transition is not None
                batch.append(transition)
                leaf_indices.append(leaf_idx)
                priorities.append(float(candidate_priorities[pos]))

        perm = np.random.permutation(len(batch))
        batch = [batch[i] for i in perm]
        indices = np.asarray([leaf_indices[i] for i in perm], dtype=np.int32)
        priorities_arr = np.asarray([priorities[i] for i in perm], dtype=np.float32)

        total = max(self.tree.total, self.eps)
        probs = priorities_arr / total
        n = float(len(self))
        is_w = (n * probs) ** (-self.beta)
        is_w /= is_w.max()
        self.beta = min(self.beta1, self.beta + self.beta_inc)

        return batch, indices, is_w.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.eps
        self._max_priority = max(self._max_priority, float(td.max()))
        for idx, t in zip(indices, td):
            self.tree.update(int(idx), float(t) ** self.alpha)


# =============================================================================
#   Rank-based PER (shared buffer + task tracking)
# =============================================================================
class RankBasedReplayBuffer(_TaskBalancedMixin):
    """Rank-based Prioritized Experience Replay.

    Priorities are converted into ranks and sampling probabilities are
    assigned according to:

        P(i) ∝ 1 / rank(i)^α

    This approach is more robust to extreme TD errors than
    proportional PER and often produces more stable replay dynamics.

    Supports:
    - rank-based prioritization
    - balanced multitask sampling
    - importance sampling correction
    """
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.7,
        beta0: float = 0.4,
        beta1: float = 1.0,
        beta_anneal_steps: int = 200_000,
        eta: float = 0.10,
        refresh_interval: int = 128,
        eps: float = 1e-12,
    ) -> None:
        assert capacity > 1
        self.capacity = int(capacity)
        self.data: list[Optional[Transition]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.valid = np.zeros(self.capacity, dtype=np.bool_)
        self.size = 0
        self.write = 0

        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.beta1 = float(beta1)
        self.beta_inc = (self.beta1 - self.beta) / max(1, int(beta_anneal_steps))
        self.eta = float(eta)
        self.refresh_interval = int(refresh_interval)
        self.eps = float(eps)
        self._max_priority = 1.0

        self._dirty = True
        self._steps_since_refresh = 0
        self._valid_indices: np.ndarray = np.array([], dtype=np.int32)
        self._pmf: np.ndarray = np.array([], dtype=np.float64)
        self._cdf: np.ndarray = np.array([], dtype=np.float64)
        self._cdf_index: np.ndarray = np.array([], dtype=np.int32)

        self._init_task_tracking()

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition, priority: Optional[float] = None) -> None:
        idx = self.write
        self.data[idx] = transition
        base = self._max_priority if priority is None else abs(float(priority))
        self.priorities[idx] = max(base, self.eps)
        self.valid[idx] = True
        self._register_index(idx, transition.task_id)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._dirty = True

    def add_episode(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(
        self,
        batch_size: int,
    ) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        if self.size < batch_size:
            return None

        self._maybe_refresh()

        batch: list[Transition] = []
        indices = np.empty(batch_size, dtype=np.int32)
        probs = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            if random.random() < self.eta or self._cdf.size == 0:
                idx = int(np.random.choice(self._valid_indices))
                p = 1.0 / float(len(self._valid_indices))
            else:
                r = random.random()
                j = int(np.searchsorted(self._cdf, r, side="left"))
                j = min(j, self._cdf.size - 1)
                idx = int(self._cdf_index[j])
                p = float(self._pmf[j])

            batch.append(self.data[idx])
            indices[i] = idx
            probs[i] = max(p, self.eps)

        n = float(len(self))
        is_w = (n * probs) ** (-self.beta)
        is_w /= is_w.max()
        self.beta = min(self.beta1, self.beta + self.beta_inc)
        return batch, indices, is_w.astype(np.float32)

    def sample_balanced(
        self,
        batch_size: int,
        task_ids: Optional[list[int]] = None,
    ) -> Optional[tuple[list[Transition], np.ndarray, np.ndarray]]:
        if self.size < batch_size:
            return None

        self._maybe_refresh()
        active_task_ids = self._resolve_task_ids(task_ids)
        counts = self._balanced_counts(batch_size, active_task_ids)

        batch: list[Transition] = []
        sampled_indices: list[int] = []
        sampled_probs: list[float] = []

        for task_id, n_draw in counts.items():
            candidate_indices = np.array(
                sorted(idx for idx in self.task_to_indices[task_id] if self.valid[idx]),
                dtype=np.int32,
            )
            if candidate_indices.size == 0:
                return None

            candidate_priorities = np.maximum(
                self.priorities[candidate_indices].astype(np.float64),
                self.eps,
            )

            # rank-based within task
            order = np.argsort(-candidate_priorities, kind="mergesort")
            ranked_indices = candidate_indices[order]
            ranks = np.arange(1, ranked_indices.size + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, self.alpha)
            pmf = weights / weights.sum()

            replace = ranked_indices.size < n_draw
            chosen_pos = np.random.choice(
                np.arange(ranked_indices.size),
                size=n_draw,
                replace=replace,
                p=pmf,
            )

            for pos in chosen_pos:
                idx = int(ranked_indices[pos])
                transition = self.data[idx]
                assert transition is not None
                batch.append(transition)
                sampled_indices.append(idx)
                sampled_probs.append(float(pmf[pos]))

        perm = np.random.permutation(len(batch))
        batch = [batch[i] for i in perm]
        indices = np.asarray([sampled_indices[i] for i in perm], dtype=np.int32)
        probs = np.asarray([sampled_probs[i] for i in perm], dtype=np.float64)

        n = float(len(self))
        is_w = (n * probs) ** (-self.beta)
        is_w /= is_w.max()
        self.beta = min(self.beta1, self.beta + self.beta_inc)

        return batch, indices, is_w.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        abs_td = np.abs(np.asarray(td_errors, dtype=np.float32))
        self.priorities[indices] = np.maximum(abs_td, self.eps)
        self._max_priority = max(self._max_priority, float(abs_td.max()))
        self._dirty = True

    def _maybe_refresh(self) -> None:
        if not self._dirty and self._steps_since_refresh < self.refresh_interval:
            self._steps_since_refresh += 1
            return

        valid_idx = np.nonzero(self.valid)[0]
        if valid_idx.size == 0:
            self._valid_indices = np.array([], dtype=np.int32)
            self._pmf = np.array([], dtype=np.float64)
            self._cdf = np.array([], dtype=np.float64)
            self._cdf_index = np.array([], dtype=np.int32)
            self._dirty = False
            self._steps_since_refresh = 0
            return

        pri = self.priorities[valid_idx]
        order = np.argsort(-pri, kind="mergesort")
        ranked_idx = valid_idx[order]
        ranks = np.arange(1, ranked_idx.size + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.alpha)
        pmf = weights / weights.sum()
        cdf = np.cumsum(pmf)

        self._valid_indices = ranked_idx.astype(np.int32)
        self._pmf = pmf
        self._cdf = cdf
        self._cdf_index = ranked_idx.astype(np.int32)

        self._dirty = False
        self._steps_since_refresh = 0