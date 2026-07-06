"""
diagnostics_utils.py — Delphos v1.2
=====================================
Pure stateless computation helpers for all diagnostics.

All functions are side-effect-free: they accept primitives / lists / numpy arrays
and return plain dicts, floats, or arrays.  No I/O, no trainer state.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# A1.1 — Reward Diagnostics
# =============================================================================

def compute_reward_stats(rewards: Sequence[float]) -> Dict[str, float]:
    """Compute aggregate reward statistics from a sequence of episode rewards.

    Args:
        rewards (Sequence[float]): A sequence of float rewards.

    Returns:
        Dict[str, float]: dict with keys: mean, std, min, max, median,
            positive_ratio, neg1_ratio, n_episodes.
    """
    if not rewards:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"), "median": float("nan"),
            "positive_ratio": float("nan"), "neg1_ratio": float("nan"),
            "n_episodes": 0,
        }
    arr = np.array(rewards, dtype=np.float64)
    n = len(arr)
    return {
        "mean":           float(np.nanmean(arr)),
        "std":            float(np.nanstd(arr)),
        "min":            float(np.nanmin(arr)),
        "max":            float(np.nanmax(arr)),
        "median":         float(np.nanmedian(arr)),
        "positive_ratio": float(np.sum(arr > 0.0) / n),
        "neg1_ratio":     float(np.sum(arr == -1.0) / n),
        "n_episodes":     int(n),
    }


def rolling_reward_stats(
    rewards: Sequence[float],
    window: int = 50,
) -> Dict[str, List[float]]:
    """Compute rolling mean and rolling std over a reward sequence.

    Args:
        rewards (Sequence[float]): Sequence of rewards.
        window (int, optional): The rolling window size. Defaults to 50.

    Returns:
        Dict[str, List[float]]: A dictionary containing 'rolling_mean' and 'rolling_std' lists
            of length len(rewards) (nan-padded for the initial window).
    """
    arr = np.array(rewards, dtype=np.float64)
    n = len(arr)
    rolling_mean = [float("nan")] * n
    rolling_std  = [float("nan")] * n
    for i in range(n):
        start = max(0, i - window + 1)
        window_slice = arr[start : i + 1]
        rolling_mean[i] = float(np.mean(window_slice))
        rolling_std[i]  = float(np.std(window_slice))
    return {"rolling_mean": rolling_mean, "rolling_std": rolling_std}


# =============================================================================
# A1.2 — Episode Diagnostics
# =============================================================================

def compute_episode_stats(
    episode_lengths: Sequence[int],
    horizon_lengths: Sequence[int],
    forced_stops: Sequence[bool],
) -> Dict[str, float]:
    """
    Aggregate episode-level statistics.

    Args:
        episode_lengths: Episode lengths.
        horizon_lengths: Episode horizons.
        forced_stops: Whether episode terminated by horizon.

    Returns:
        Dictionary of summary statistics.
    """

    if not episode_lengths:
        return {}

    L = np.asarray(episode_lengths, dtype=np.float64)
    H = np.asarray(horizon_lengths, dtype=np.float64)
    F = np.asarray(forced_stops, dtype=np.float64)

    horizon_ratio = L / np.maximum(H, 1.0)

    return {
        "mean_L": float(np.mean(L)),
        "std_L": float(np.std(L)),
        "min_L": float(np.min(L)),
        "max_L": float(np.max(L)),
        "mean_horizon_ratio": float(np.mean(horizon_ratio)),
        "forced_stop_ratio": float(np.mean(F)),
        "n_episodes": int(len(L)),
    }


# =============================================================================
# A1.3 — Exploration / Action Entropy Diagnostics
# =============================================================================

def compute_action_entropy(action_counts: Dict[int, int]) -> float:
    """Compute the Shannon entropy (nats) of the action frequency distribution.

    Args:
        action_counts (Dict[int, int]): A mapping from action index to frequency count.

    Returns:
        float: The Shannon entropy. Returns 0.0 if no actions have been taken.
    """
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in action_counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def compute_action_histogram(
    action_counts: Dict[int, int],
    num_catalogue_actions: int,
) -> Dict[str, Any]:
    """Build action usage diagnostics from a flat action-frequency counter.

    Args:
        action_counts (Dict[int, int]): Dictionary of action frequencies.
        num_catalogue_actions (int): Total number of actions available in the catalogue.

    Returns:
        Dict[str, Any]: A dictionary containing: entropy, pct_catalogue_explored, 
            top5_actions, dead_action_count, dead_action_pct, dominance_ratio.
    """
    total = sum(action_counts.values())
    n_explored = len([v for v in action_counts.values() if v > 0])
    pct_explored = n_explored / max(num_catalogue_actions, 1)
    entropy = compute_action_entropy(action_counts)

    # Dominance: fraction of calls taken by the single most-used action
    dominance_ratio = 0.0
    if total > 0 and action_counts:
        dominance_ratio = max(action_counts.values()) / total

    dead_count = num_catalogue_actions - n_explored
    dead_pct   = dead_count / max(num_catalogue_actions, 1)

    top5 = sorted(action_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "entropy":              entropy,
        "pct_catalogue_explored": float(pct_explored),
        "n_actions_explored":   int(n_explored),
        "dead_action_count":    int(dead_count),
        "dead_action_pct":      float(dead_pct),
        "dominance_ratio":      float(dominance_ratio),
        "top5_actions":         top5,
        "total_action_calls":   int(total),
    }


# =============================================================================
# A2.1 — Specification Diversity
# =============================================================================

def compute_spec_diversity(
    specs: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute diversity metrics over a list of specification strings (apollo keys).

    Args:
        specs (Sequence[str]): A sequence of specification strings.
        top_k (int, optional): Number of top specifications to return. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing: n_unique, entropy, top_k_specs, repeated_ratio.
    """
    if not specs:
        return {
            "n_unique": 0,
            "entropy": 0.0,
            "top_k_specs": [],
            "repeated_ratio": 0.0,
            "novelty_ratio": 0.0,
            "n_total": 0,
        }

    counter = Counter(specs)
    n_total  = len(specs)
    n_unique = len(counter)

    probs = np.array([v / n_total for v in counter.values()], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    top_k_specs = counter.most_common(top_k)
    repeated_ratio = float((n_total - n_unique) / n_total)
    novelty_ratio = float(n_unique / n_total)

    return {
        "n_unique":      int(n_unique),
        "entropy":       float(entropy),
        "top_k_specs":   top_k_specs,
        "repeated_ratio": float(repeated_ratio),
        "novelty_ratio": float(novelty_ratio),
        "n_total":       int(n_total),
    }


def _spec_to_token_set(spec_key: str) -> set:
    """Parse an apollo spec key into a set of segment tokens.

    Args:
        spec_key (str): The specification string key.

    Returns:
        set: A set of string tokens.
    """
    return set(spec_key.split("_"))


def compute_spec_hamming_distance(specs: Sequence[str]) -> float:
    """Compute average pairwise symmetric-difference distance between spec keys.

    Each spec is treated as a bag of underscore-delimited tokens (segments).
    Distance = |A △ B| / |A ∪ B|.

    Args:
        specs (Sequence[str]): A sequence of specification strings.

    Returns:
        float: The average distance. Returns 0.0 if fewer than 2 specs.
    """
    if len(specs) < 2:
        return 0.0

    token_sets = [_spec_to_token_set(s) for s in specs]
    dists = []
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            sym_diff = a ^ b
            d = len(sym_diff) / max(len(union), 1)
            dists.append(d)

    return float(np.mean(dists)) if dists else 0.0


# =============================================================================
# A2.2 — Trajectory Diversity
# =============================================================================

def trajectory_key(action_sequence: Sequence[int]) -> str:
    """Convert an ordered action sequence to a canonical string key.

    Args:
        action_sequence (Sequence[int]): Sequence of integer action indices.

    Returns:
        str: Underscore-separated string key representing the trajectory.
    """
    return "_".join(str(a) for a in action_sequence)


def compute_trajectory_diversity(trajectories: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Compute diversity metrics over a collection of action trajectories.

    Args:
        trajectories (Sequence[Sequence[int]]): A sequence of trajectories, 
            where each trajectory is a sequence of global action indices.

    Returns:
        Dict[str, Any]: A dictionary containing trajectory diversity metrics.
    """
    if not trajectories:
        return {"n_unique": 0, "entropy": 0.0, "repeated_ratio": 0.0}

    keys = [trajectory_key(t) for t in trajectories]
    counter = Counter(keys)
    n_total  = len(keys)
    n_unique = len(counter)

    probs = np.array([v / n_total for v in counter.values()], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    repeated_ratio = float((n_total - n_unique) / n_total)

    return {
        "n_unique":       int(n_unique),
        "entropy":        float(entropy),
        "repeated_ratio": float(repeated_ratio),
        "n_total":        int(n_total),
    }


# =============================================================================
# A2.3 — Embedding Diversity
# =============================================================================

def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute diversity statistics over a 2-D embedding matrix (N × D).

    Returns export-ready dict. No plotting is performed.

    Args:
        embeddings (np.ndarray): The N × D embedding matrix.

    Returns:
        Dict[str, Any]: dictionary with: mean_cosine_sim, std_cosine_sim, mean_l2_norm,
            mean_variance_per_dim, embedding_matrix (for PCA/UMAP).
    """
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return {
            "mean_cosine_sim": float("nan"),
            "std_cosine_sim":  float("nan"),
            "mean_l2_norm":    float("nan"),
            "mean_variance_per_dim": float("nan"),
            "embedding_matrix": embeddings,
        }

    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms

    # Sample up to 500 pairs for efficiency
    n = embeddings.shape[0]
    n_sample = min(n, 500)
    idx = np.random.choice(n, size=n_sample, replace=False) if n > n_sample else np.arange(n)
    sub = normed[idx]

    # Cosine similarity matrix
    sim_matrix = sub @ sub.T
    # Extract upper triangle (exclude diagonal)
    upper = sim_matrix[np.triu_indices(n_sample, k=1)]

    return {
        "mean_cosine_sim":       float(np.mean(upper)),
        "std_cosine_sim":        float(np.std(upper)),
        "mean_l2_norm":          float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "mean_variance_per_dim": float(np.mean(np.var(embeddings, axis=0))),
        "embedding_matrix":      embeddings,   # caller can pass to PCA/UMAP
    }


# =============================================================================
# A1.5 — Q-Learning Diagnostics
# =============================================================================

def compute_q_stats(q_values: np.ndarray, threshold: float = -1e4) -> Dict[str, float]:
    """Aggregate Q-value statistics from a flat or batched array.

    Args:
        q_values (np.ndarray): 1-D or 2-D array of Q values.
        threshold (float, optional): Q-values below this are counted as "dead" actions. Defaults to -1e4.

    Returns:
        Dict[str, float]: Dictionary of computed Q statistics.
    """
    arr = np.asarray(q_values, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0:
        return {}

    dead_pct = float(np.sum(arr < threshold) / n)
    dominance = float((arr == arr.max()).sum() / n)

    return {
        "q_mean":          float(np.mean(arr)),
        "q_std":           float(np.std(arr)),
        "q_min":           float(np.min(arr)),
        "q_max":           float(np.max(arr)),
        "dead_action_pct": dead_pct,
        "dominance_ratio": dominance,
    }


def compute_update_stats(
    loss: float,
    td_errors: np.ndarray,
    q_values_sa: np.ndarray,
    target_values: np.ndarray,
    grad_norm: float,
    param_update_magnitude: float,
) -> Dict[str, float]:
    """Aggregate per-update statistics for one gradient step.

    Args:
        loss (float): The loss value for the update step.
        td_errors (np.ndarray): Array of temporal difference errors.
        q_values_sa (np.ndarray): Array of Q-values for the state-action pairs.
        target_values (np.ndarray): Array of target Q-values.
        grad_norm (float): The computed gradient norm.
        param_update_magnitude (float): The magnitude of the parameter update.

    Returns:
        Dict[str, float]: Dictionary containing computed update statistics.
    """
    td = np.asarray(td_errors, dtype=np.float64)
    q  = np.asarray(q_values_sa, dtype=np.float64)
    tgt = np.asarray(target_values, dtype=np.float64)

    return {
        "loss":                   float(loss),
        "td_error_mean":          float(np.mean(td)),
        "td_error_max":           float(np.max(td)),
        "td_error_std":           float(np.std(td)),
        "q_mean":                 float(np.mean(q)),
        "q_std":                  float(np.std(q)),
        "q_min":                  float(np.min(q)),
        "q_max":                  float(np.max(q)),
        "target_q_mean":          float(np.mean(tgt)),
        "target_q_std":           float(np.std(tgt)),
        "grad_norm":              float(grad_norm),
        "param_update_magnitude": float(param_update_magnitude),
    }


# =============================================================================
# A1.4 — Replay Buffer Diagnostics
# =============================================================================

def compute_replay_stats(buffer) -> Dict[str, Any]:
    """Extract diagnostics from any replay buffer with the _TaskBalancedMixin API.

    Works for UniformReplayBuffer, PrioritizedReplayBuffer, RankBasedReplayBuffer.

    Args:
        buffer: The replay buffer instance.

    Returns:
        Dict[str, Any]: Dictionary containing buffer statistics.
    """
    result: Dict[str, Any] = {
        "buffer_size":       int(len(buffer)),
        "buffer_capacity":   int(buffer.capacity),
        "fill_ratio":        float(len(buffer) / max(buffer.capacity, 1)),
        "transitions_per_task": buffer.transitions_per_task(),
    }

    # Collect reward and terminal statistics from available data
    rewards = []
    terminal_count = 0
    unique_actions: set = set()
    n_samples = min(len(buffer), 2000)

    if n_samples > 0:
        # Access the raw data list (common to all buffer types)
        data_attr = getattr(buffer, "data", None)
        if data_attr is None:
            # PrioritizedReplayBuffer uses tree.data
            tree = getattr(buffer, "tree", None)
            data_attr = tree.data if tree is not None else []

        valid_transitions = [t for t in data_attr if t is not None]
        sample = valid_transitions[:n_samples]

        for t in sample:
            rewards.append(float(t.reward))
            if t.done:
                terminal_count += 1
            unique_actions.add(int(t.action_index))

        n = len(sample)
        arr_r = np.array(rewards, dtype=np.float64)
        result.update({
            "reward_mean":      float(np.mean(arr_r)),
            "reward_std":       float(np.std(arr_r)),
            "reward_min":       float(np.min(arr_r)),
            "reward_max":       float(np.max(arr_r)),
            "terminal_ratio":   float(terminal_count / max(n, 1)),
            "n_unique_actions": int(len(unique_actions)),
            "sampled_n":        int(n),
        })

    return result
