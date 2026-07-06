"""
diagnostics/transfer.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Transfer-learning diagnostics and analysis utilities.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from transfer.transfer_results import TransferResult


# =============================================================================
# Internal
# =============================================================================

def _results_to_dataframe(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Convert TransferResult objects into a dataframe.
    """
    if len(results) == 0:
        return pd.DataFrame()

    rows = []

    for result in results:

        metadata = (
            result.metadata
            if result.metadata is not None
            else {}
        )

        rows.append(
            {
                "source_task_id": result.source_task_id,
                "source_task_name": result.source_task_name,

                "task_id": result.task_id,
                "task_name": result.task_name,

                "experiment": result.experiment,

                "reward": result.reward,
                "initial_reward": result.initial_reward,
                "improvement": result.improvement,

                "adaptation_rounds": result.adaptation_rounds,
                "episode_length": result.episode_length,

                "specification_key": result.specification_key,

                "adaptation_mode": metadata.get("adaptation_mode"),
                "search_strategy": metadata.get("search_strategy"),

                "epsilon": metadata.get("epsilon"),
                "temperature": metadata.get("temperature"),
                "top_k": metadata.get("top_k"),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Overall Transfer Summary
# =============================================================================

def summarize_transfer_results(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Aggregate transfer-learning results by experiment.

    Returns
    -------
    experiment
    reward_mean
    reward_std
    reward_max
    reward_min
    improvement_mean
    improvement_std
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for experiment, group in df.groupby("experiment"):

        rows.append(
            {
                "experiment": str(experiment),
                "n_runs": int(len(group)),
                "reward_mean": float(group["reward"].mean()),
                "reward_std": float(group["reward"].std()),
                "reward_max": float(group["reward"].max()),
                "reward_min": float(group["reward"].min()),
                "improvement_mean": float(group["improvement"].mean()),
                "improvement_std": float(group["improvement"].std()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Adaptation Modes
# =============================================================================

def compare_adaptation_modes(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Compare:

    - inference
    - policy_only
    - encoder_only
    - fine_tune_all
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for mode, group in df.groupby("adaptation_mode"):

        rows.append(
            {
                "adaptation_mode": str(mode),
                "n_runs": int(len(group)),
                "reward_mean": float(group["reward"].mean()),
                "reward_std": float(group["reward"].std()),
                "reward_max": float(group["reward"].max()),
                "improvement_mean": float(group["improvement"].mean()),
                "adaptation_rounds_mean": float(
                    group["adaptation_rounds"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Search Strategies
# =============================================================================

def compare_search_strategies(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Compare:

    - greedy
    - stochastic
    - boltzmann
    - topk
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for strategy, group in df.groupby("search_strategy"):

        rows.append(
            {
                "search_strategy": str(strategy),
                "n_runs": int(len(group)),
                "reward_mean": float(group["reward"].mean()),
                "reward_std": float(group["reward"].std()),
                "reward_max": float(group["reward"].max()),
                "reward_min": float(group["reward"].min()),
                "improvement_mean": float(group["improvement"].mean()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Adaptation Efficiency
# =============================================================================

def adaptation_efficiency(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Improvement per adaptation round.

    Useful for:

    - Few-shot transfer
    - Sample efficiency analysis
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["improvement_per_round"] = (
        df["improvement"]
        /
        np.maximum(df["adaptation_rounds"], 1)
    )

    return df[
        [
            "task_id",
            "task_name",
            "experiment",
            "adaptation_mode",
            "search_strategy",
            "adaptation_rounds",
            "improvement",
            "improvement_per_round",
        ]
    ]


# =============================================================================
# Transfer Matrix
# =============================================================================
def transfer_matrix(
    results: Sequence[TransferResult],
) -> pd.DataFrame:

    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    return (
        df.pivot_table(
            index="source_task_name",
            columns="task_name",
            values="reward",
            aggfunc="mean",
        )
        .sort_index()
    )


def transfer_improvement_matrix(
    results: Sequence[TransferResult],
) -> pd.DataFrame:

    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    return (
        df.pivot_table(
            index="source_task_name",
            columns="task_name",
            values="improvement",
            aggfunc="mean",
        )
        .sort_index()
    )


# =============================================================================
# Best Transfer Results
# =============================================================================

def best_transfer_results(
    results: Sequence[TransferResult],
    n: int = 20,
) -> pd.DataFrame:
    """
    Top-N transfer experiments.
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    return (
        df.sort_values(
            "reward",
            ascending=False,
        )
        .head(int(n))
        .reset_index(drop=True)
    )


# =============================================================================
# Worst Transfer Results
# =============================================================================

def worst_transfer_results(
    results: Sequence[TransferResult],
    n: int = 20,
) -> pd.DataFrame:
    """
    Worst-N transfer experiments.
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    return (
        df.sort_values(
            "reward",
            ascending=True,
        )
        .head(int(n))
        .reset_index(drop=True)
    )


# =============================================================================
# Zero-shot vs Few-shot
# =============================================================================

def compare_zero_vs_few_shot(
    results: Sequence[TransferResult],
) -> pd.DataFrame:
    """
    Compare:

    - adaptation_mode=inference
    - adaptation_mode!=inference
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["setting"] = np.where(
        df["adaptation_mode"] == "inference",
        "zero_shot",
        "few_shot",
    )

    rows = []

    for setting, group in df.groupby("setting"):

        rows.append(
            {
                "setting": setting,
                "n_runs": int(len(group)),
                "reward_mean": float(group["reward"].mean()),
                "reward_std": float(group["reward"].std()),
                "reward_max": float(group["reward"].max()),
                "improvement_mean": float(group["improvement"].mean()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Compact Summary
# =============================================================================

def transfer_summary(
    results: Sequence[TransferResult],
) -> dict:
    """
    High-level transfer summary.
    """
    df = _results_to_dataframe(results)

    if df.empty:
        return {}

    return {
        "n_runs": int(len(df)),
        "reward_mean": float(df["reward"].mean()),
        "reward_std": float(df["reward"].std()),
        "reward_max": float(df["reward"].max()),
        "reward_min": float(df["reward"].min()),
        "improvement_mean": float(df["improvement"].mean()),
        "improvement_std": float(df["improvement"].std()),
        "n_tasks": int(df["task_id"].nunique()),
        "n_experiments": int(df["experiment"].nunique()),
    }