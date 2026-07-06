"""
diagnostics_logger.py — Delphos v1.2
======================================
Stateful logger that accumulates per-episode and per-update rows in memory,
then hands them to diagnostics_summary.py for aggregation and CSV export.

Usage (opt-in)
--------------
    from diagnostics.diagnostics_logger import DiagnosticsLogger

    logger = DiagnosticsLogger(cap=100_000)
    trainer.diag_logger = logger          # attach to trainer

After training:
    from diagnostics.diagnostics_summary import export_all_csvs
    export_all_csvs(logger, run_dir=Path(trainer.subfolder))
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from diagnostics.diagnostics_utils import compute_action_entropy, trajectory_key


# =============================================================================
# DiagnosticsLogger
# =============================================================================

class DiagnosticsLogger:

    def __init__(self, cap: int = 100_000,):
        self.cap = int(cap)
        self._episode_rows = []
        self._update_rows = []
        self._seen_specs = {}
        self._embeddings = {}

    def _ensure_task(self, task_id: int,):
        if task_id not in self._seen_specs:
            self._seen_specs[task_id] = set()
            self._embeddings[task_id] = []

    def log_episode(
        self,
        task_id: int,
        task_name: str,
        episode: int,
        round_idx: int,
        epsilon: float,
        reward: float,
        L_e: int,
        H_tau: int,
        forced_stop: bool,
        spec_key: str,
        action_sequence: list[int],
        valid_actions_per_step: list[int],
        unique_specifications: int,
        unique_actions: int,
        runtime_seconds: float,
        z_states: list[np.ndarray] | None = None,
    ):
        self._ensure_task(task_id)
        is_repeated_spec = (spec_key in self._seen_specs[task_id])
        self._seen_specs[task_id].add(spec_key)

        if z_states:
            final_embedding = np.asarray(z_states[-1]).flatten()
            self._embeddings[task_id].append(final_embedding)
        else:
            final_embedding = None

        row = {
            "task_id": task_id,
            "task_name": task_name,
            "episode": episode,
            "round_idx": round_idx,
            "epsilon": epsilon,
            "reward": reward,
            "episode_length": L_e,
            "horizon": H_tau,
            "horizon_ratio": (L_e / max(H_tau, 1)),
            "forced_stop": forced_stop,
            "specification_key": spec_key,
            "action_sequence": action_sequence,
            "mean_valid_actions": np.mean(valid_actions_per_step) if len(valid_actions_per_step) > 0 else np.nan,
            "unique_specifications": unique_specifications,
            "unique_actions": unique_actions,
            "runtime_seconds": runtime_seconds,
            "is_repeated_spec": is_repeated_spec,
            "n_unique_specs": len(self._seen_specs[task_id]),

        }
        self._episode_rows.append(row)
        if len(self._episode_rows) > self.cap:
            self._episode_rows.pop(0)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat(timespec="seconds") 

    def _cap_list(self, lst: list) -> None:
        excess = len(lst) - self.cap
        if excess > 0:
            del lst[:excess]
    
    # ------------------------------------------------------------------
    # A1.5 — Update (gradient step) logging
    # ------------------------------------------------------------------

    def log_update(
        self,
        episode:               int,
        round_idx:             int,
        epsilon:               float,
        loss:                  float,
        td_errors:             np.ndarray,
        q_values_sa:           np.ndarray,
        target_values:         np.ndarray,
        grad_norm:             float,
        param_update_magnitude: float,
    ) -> None:
        """Record one gradient update step (A1.5).

        Args:
            episode (int): The current episode number.
            round_idx (int): The current round index.
            epsilon (float): Epsilon exploration parameter.
            loss (float): Loss from update.
            td_errors (np.ndarray): TD errors.
            q_values_sa (np.ndarray): Q-values.
            target_values (np.ndarray): Target Q-values.
            grad_norm (float): Gradient norm.
            param_update_magnitude (float): Parameter update magnitude.
        """
        td  = np.asarray(td_errors, dtype=np.float64)
        q   = np.asarray(q_values_sa, dtype=np.float64)
        tgt = np.asarray(target_values, dtype=np.float64)

        row: Dict[str, Any] = {
            "timestamp":              self._now(),
            "episode":                int(episode),
            "round_idx":              int(round_idx),
            "epsilon":                float(epsilon),
            # Loss & TD
            "loss":                   float(loss),
            "td_error_mean":          float(np.mean(td)),
            "td_error_max":           float(np.max(td)),
            "td_error_std":           float(np.std(td)),
            "abs_td_error_mean":      float(np.mean(np.abs(td))),
            "abs_td_error_max":       float(np.max(np.abs(td))),
            "abs_td_error_std":       float(np.std(np.abs(td))),
            # Q statistics
            "q_mean":                 float(np.mean(q)),
            "q_std":                  float(np.std(q)),
            "q_min":                  float(np.min(q)),
            "q_max":                  float(np.max(q)),
            # Target Q
            "target_q_mean":          float(np.mean(tgt)),
            "target_q_std":           float(np.std(tgt)),
            # Gradient / parameter dynamics
            "grad_norm":              float(grad_norm),
            "param_update_magnitude": float(param_update_magnitude),
        }

        self._update_rows.append(row)
        self._cap_list(self._update_rows)

    # ------------------------------------------------------------------
    # A3 — Inference / transfer episode logging
    # ------------------------------------------------------------------

    def log_inference(
        self,
        episode:         int,
        task_id:         int,
        task_name:       str,
        reward:          float,
        L_e:             int,
        n_terms:         int,
        spec_key:        str,
        mode:            str,
        action_sequence: Optional[Sequence[int]] = None,
    ) -> None:
        """Record one inference / transfer episode.

        Args:
            episode (int): The episode number.
            task_id (int): ID of the task.
            task_name (str): Name of the task.
            reward (float): Final reward of episode.
            L_e (int): Length of episode.
            n_terms (int): Number of terms in spec.
            spec_key (str): Key representing the final state specification.
            mode (str): Name of inference mode (e.g. "greedy").
            action_sequence (Optional[Sequence[int]], optional): List of actions taken. Defaults to None.
        """
        self._ensure_task(task_id)

        traj_key = trajectory_key(action_sequence) if action_sequence else ""
        is_repeated_spec = spec_key in self._seen_specs[task_id]
        self._seen_specs[task_id].add(spec_key)

        row: Dict[str, Any] = {
            "timestamp":          self._now(),
            "episode":            int(episode),
            "task_id":            int(task_id),
            "task_name":          str(task_name),
            "mode":               str(mode),
            "reward":             float(reward),
            "L_e":                int(L_e),
            "n_terms":            int(n_terms),
            "spec_key":           str(spec_key),
            "trajectory_key":     str(traj_key),
            "is_repeated_spec":   bool(is_repeated_spec),
            "n_unique_specs":     int(len(self._seen_specs[task_id])),
        }

        self._inference_rows.append(row)
        self._cap_list(self._inference_rows)

    # ------------------------------------------------------------------
    # DataFrame accessors
    # ------------------------------------------------------------------

    @property
    def episode_df(self) -> pd.DataFrame:
        """Return accumulated episode rows as a DataFrame."""
        return pd.DataFrame(self._episode_rows)

    @property
    def update_df(self) -> pd.DataFrame:
        """Return accumulated update rows as a DataFrame."""
        return pd.DataFrame(self._update_rows)

    @property
    def inference_df(self) -> pd.DataFrame:
        """Return accumulated inference/transfer rows as a DataFrame."""
        return pd.DataFrame(self._inference_rows)

    # ------------------------------------------------------------------
    # Embedding accessors
    # ------------------------------------------------------------------

    def get_embeddings(self, task_id: int) -> Optional[np.ndarray]:
        """Return stacked embedding matrix for a given task_id (N × D), or None.
        
        Use for downstream PCA / UMAP / t-SNE.

        Args:
            task_id (int): The target task ID.

        Returns:
            Optional[np.ndarray]: A 2D numpy array of stacked embeddings, or None.
        """
        embs = self._embeddings.get(task_id)
        if not embs:
            return None
        return np.stack(embs, axis=0)

    def get_action_counts(self, task_id: int) -> Counter:
        """Return cumulative action frequency counter for a task."""
        return self._action_counts.get(task_id, Counter())

    def get_seen_specs(self, task_id: int) -> set:
        """Return the set of unique specification keys seen for a task."""
        return self._seen_specs.get(task_id, set())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated rows (useful between phases)."""
        self._episode_rows.clear()
        self._update_rows.clear()
        self._inference_rows.clear()
        self._seen_specs.clear()
        self._seen_trajectories.clear()
        self._action_counts.clear()
        self._embeddings.clear()
