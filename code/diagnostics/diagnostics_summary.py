"""
diagnostics_summary.py — Delphos v1.2
=====================================
Aggregation and CSV export utilities.

Exports
-------
per_episode_diagnostics.csv
per_update_diagnostics.csv
per_task_summary.csv
replay_diagnostics.csv
diversity_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from diagnostics.diagnostics_utils import (
    compute_reward_stats,
    compute_episode_stats,
    compute_spec_diversity,
    compute_spec_hamming_distance,
    compute_embedding_stats,
    compute_replay_stats,
)


# =============================================================================
# IO Helpers
# =============================================================================
def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _safe_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    path.write_text(json.dumps(_clean(data), indent=2), encoding="utf-8")

# =============================================================================
# Per-task summary
# =============================================================================

def compute_per_task_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    rows = []
    for task_id, group in episode_df.groupby("task_id"):
        task_name = group["task_name"].iloc[0]
        rewards = group["reward"].tolist()
        reward_stats = compute_reward_stats(rewards)
        episode_stats = compute_episode_stats(episode_lengths=group["episode_length"].tolist(), horizon_lengths=group["horizon"].tolist(), forced_stops=group["forced_stop"].tolist())
        spec_keys = group["specification_key"].astype(str).tolist()
        spec_diversity = compute_spec_diversity(spec_keys)
        row = {
            "task_id": int(task_id),
            "task_name": str(task_name),
            "n_episodes": int(len(group)),
            "reward_mean": reward_stats["mean"],
            "reward_std": reward_stats["std"],
            "reward_min": reward_stats["min"],
            "reward_max": reward_stats["max"],
            "reward_median": reward_stats["median"],
            "positive_reward_ratio": reward_stats["positive_ratio"],
            "neg1_ratio": reward_stats["neg1_ratio"],
            "mean_episode_length": episode_stats.get("mean_L"),
            "std_episode_length": episode_stats.get("std_L"),
            "mean_horizon_ratio": episode_stats.get("mean_horizon_ratio"),
            "forced_stop_ratio": episode_stats.get("forced_stop_ratio"),
            "mean_valid_actions": float(group["mean_valid_actions"].mean()),
            "mean_unique_specifications": float(group["unique_specifications"].mean()),
            "mean_unique_actions": float(group["unique_actions"].mean()),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            "n_unique_specs": spec_diversity["n_unique"],
            "spec_entropy": spec_diversity["entropy"],
            "repeated_spec_ratio": float(group["is_repeated_spec"].mean()),
        }
        rows.append(row)
        
    return pd.DataFrame(rows)


# =============================================================================
# Replay Buffer Diagnostics
# =============================================================================

def compute_replay_diagnostics(buffer, run_dir: Optional[Path] = None,) -> pd.DataFrame:
    stats = compute_replay_stats(buffer)
    transitions_per_task = stats.pop("transitions_per_task", {},)

    rows = []
    for k, v in stats.items():
        rows.append({"metric": k, "value": v})
    for task_id, count in transitions_per_task.items():
        rows.append({"metric": f"transitions_task_{task_id}", "value": count})

    df = pd.DataFrame(rows)
    if run_dir is not None:
        _safe_to_csv(df, Path(run_dir) / "replay_diagnostics.csv",)
    return df


# =============================================================================
# Diversity Summary
# =============================================================================

def compute_diversity_summary(logger, task_name_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    summary = {}
    episode_df = logger.episode_df
    if episode_df.empty:
        return summary
  
    for task_id, group in episode_df.groupby("task_id"):
        task_name = (task_name_map or {}).get(int(task_id), str(task_id),)
        spec_keys = (group["specification_key"].astype(str).tolist())
        spec_div = compute_spec_diversity(spec_keys)
        spec_div.pop("top_k_specs", None)
        spec_div["avg_pairwise_hamming"] = compute_spec_hamming_distance(spec_keys)
        emb_matrix = logger.get_embeddings(int(task_id))
        if emb_matrix is not None:
            emb_stats = compute_embedding_stats(emb_matrix)
            emb_stats.pop("embedding_matrix", None)
        else:
            emb_stats = {}
        summary[task_name] = {
            "spec_diversity": spec_div,
            "embedding_stats": emb_stats,
        }
    return summary


# =============================================================================
# Export Everything
# =============================================================================

def export_all_csvs(logger, run_dir: Path, buffer=None, task_name_map: Optional[Dict[int, str]] = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True,)

    episode_df = logger.episode_df
    if not episode_df.empty:
        _safe_to_csv(episode_df, run_dir / "per_episode_diagnostics.csv")

    update_df = logger.update_df
    if not update_df.empty:
        _safe_to_csv(update_df, run_dir / "per_update_diagnostics.csv")

    if not episode_df.empty:
        task_summary = compute_per_task_summary(episode_df)
        if not task_summary.empty:
            _safe_to_csv(task_summary, run_dir / "per_task_summary.csv")

    if buffer is not None:
        compute_replay_diagnostics(buffer=buffer, run_dir=run_dir)

    diversity = compute_diversity_summary(logger, task_name_map=task_name_map)
    if diversity:
        _safe_json(diversity, run_dir / "diversity_summary.json")

def summarize_training(logger, run_dir: Path, task_name_map: Optional[ Dict[int, str] ] = None, buffer=None) -> Dict[str, Any]:
    export_all_csvs(logger=logger, run_dir=run_dir, buffer=buffer, task_name_map=task_name_map,)
    episode_df = logger.episode_df
    update_df = logger.update_df
    summary = {
        "n_episodes_logged": int(len(episode_df)),
        "n_updates_logged": int(len(update_df)),
    }

    if (not episode_df.empty and "reward" in episode_df.columns):
        summary["overall_reward"] = compute_reward_stats(episode_df["reward"].tolist())

    if (not update_df.empty and "loss" in update_df.columns):
        tail = update_df.tail(100)
        summary["final_loss_mean"] = float(tail["loss"].mean())
        summary["final_loss_std"] = float(tail["loss"].std())

        if "abs_td_error_mean" in tail.columns:
            summary["final_abs_td_error"] = float(tail["abs_td_error_mean"].mean())

        if "grad_norm" in tail.columns:
            summary["final_grad_norm"] = float(tail["grad_norm"].mean())

        if "q_mean" in tail.columns:
            summary["final_q_mean"] = float(tail["q_mean"].mean())
    return summary