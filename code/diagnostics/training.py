"""
diagnostics/training.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Training diagnostics and analysis utilities for Delphos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from diagnostics.diagnostics_summary import compute_per_task_summary
from diagnostics.diagnostics_utils import rolling_reward_stats, compute_replay_stats


# =============================================================================
# Reward Learning
# =============================================================================

def reward_learning_curve(logger) -> pd.DataFrame:
    """
    Build reward learning curve diagnostics.

    Returns a dataframe containing:
        - episode
        - reward
        - rolling_mean
        - rolling_std

    Parameters
    ----------
    logger : DiagnosticsLogger

    Returns
    -------
    pd.DataFrame
    """
    df = logger.episode_df

    if df.empty:
        return pd.DataFrame(
            columns=[
                "episode",
                "reward",
                "rolling_mean",
                "rolling_std",
            ]
        )

    rewards = df["reward"].tolist()

    rolling = rolling_reward_stats(rewards)

    result = pd.DataFrame(
        {
            "episode": np.arange(len(rewards)),
            "reward": rewards,
            "rolling_mean": rolling["rolling_mean"],
            "rolling_std": rolling["rolling_std"],
        }
    )

    return result


# =============================================================================
# Exploration Diagnostics
# =============================================================================

def exploration_curve(logger) -> pd.DataFrame:
    """
    Track exploration behaviour over training.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    episode
    action_entropy
    mean_valid_actions
    n_unique_specs
    n_unique_trajs
    """
    df = logger.episode_df

    if df.empty:
        return pd.DataFrame(
            columns=[
                "episode",
                "action_entropy",
                "mean_valid_actions",
                "n_unique_specs",
                "n_unique_trajs",
            ]
        )

    columns = [
        "episode",
        "action_entropy",
        "mean_valid_actions",
        "n_unique_specs",
        "n_unique_trajs",
    ]

    existing_columns = [c for c in columns if c in df.columns]

    return df[existing_columns].copy()


# =============================================================================
# Replay Buffer Diagnostics
# =============================================================================

def replay_summary(buffer) -> dict[str, Any]:
    """
    Compute replay-buffer diagnostics.

    Parameters
    ----------
    buffer : ReplayBuffer

    Returns
    -------
    dict
    """
    return compute_replay_stats(buffer)


# =============================================================================
# Q-Learning Diagnostics
# =============================================================================

def q_learning_curve(logger) -> pd.DataFrame:
    """
    Extract Q-learning diagnostics over updates.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    episode
    loss
    td_error_mean
    grad_norm
    q_mean
    target_q_mean
    """
    df = logger.update_df

    if df.empty:
        return pd.DataFrame(
            columns=[
                "episode",
                "loss",
                "td_error_mean",
                "grad_norm",
                "q_mean",
                "target_q_mean",
            ]
        )

    columns = [
        "episode",
        "loss",
        "td_error_mean",
        "grad_norm",
        "q_mean",
        "target_q_mean",
    ]

    existing_columns = [c for c in columns if c in df.columns]

    return df[existing_columns].copy()


# =============================================================================
# Task-Level Diagnostics
# =============================================================================

def task_performance_summary(logger) -> pd.DataFrame:
    """
    Aggregate performance per task.

    Parameters
    ----------
    logger : DiagnosticsLogger

    Returns
    -------
    pd.DataFrame
    """
    return compute_per_task_summary(logger.episode_df)


# =============================================================================
# Convenience Summary
# =============================================================================

def training_summary(logger, buffer=None) -> dict[str, Any]:
    """
    High-level training diagnostics summary.

    Parameters
    ----------
    logger : DiagnosticsLogger
    buffer : ReplayBuffer | None

    Returns
    -------
    dict
    """
    summary: dict[str, Any] = {}

    reward_df = reward_learning_curve(logger)

    if not reward_df.empty:
        summary["final_reward_mean"] = float(
            reward_df["rolling_mean"].dropna().iloc[-1]
        )

    update_df = q_learning_curve(logger)

    if not update_df.empty:
        summary["final_loss"] = float(
            update_df["loss"].dropna().iloc[-1]
        )

    task_df = task_performance_summary(logger)

    if not task_df.empty:
        summary["n_tasks"] = int(task_df["task_id"].nunique())
        summary["reward_mean"] = float(task_df["reward_mean"].mean())

    if buffer is not None:
        summary["replay"] = replay_summary(buffer)

    summary["n_episodes"] = int(len(logger.episode_df))
    summary["n_updates"] = int(len(logger.update_df))

    return summary


# =============================================================================
# Best Episodes
# =============================================================================

def best_episodes(
    logger,
    n: int = 20,
) -> pd.DataFrame:
    """
    Return top-N reward episodes.

    Parameters
    ----------
    logger : DiagnosticsLogger
    n : int

    Returns
    -------
    pd.DataFrame
    """
    df = logger.episode_df

    if df.empty:
        return pd.DataFrame()

    return (
        df.sort_values("reward", ascending=False)
        .head(int(n))
        .reset_index(drop=True)
    )


# =============================================================================
# Hardest Tasks
# =============================================================================

def hardest_tasks(logger) -> pd.DataFrame:
    """
    Rank tasks by average reward.

    Lower reward -> harder task.

    Parameters
    ----------
    logger : DiagnosticsLogger

    Returns
    -------
    pd.DataFrame
    """
    df = task_performance_summary(logger)

    if df.empty:
        return df

    return (
        df.sort_values("reward_mean", ascending=True)
        .reset_index(drop=True)
    )


# =============================================================================
# Easiest Tasks
# =============================================================================

def easiest_tasks(logger) -> pd.DataFrame:
    """
    Rank tasks by average reward.

    Higher reward -> easier task.

    Parameters
    ----------
    logger : DiagnosticsLogger

    Returns
    -------
    pd.DataFrame
    """
    df = task_performance_summary(logger)

    if df.empty:
        return df

    return (
        df.sort_values("reward_mean", ascending=False)
        .reset_index(drop=True)
    )