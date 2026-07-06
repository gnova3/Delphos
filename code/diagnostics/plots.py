"""
diagnostics/plots.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Visualization utilities for Delphos diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from diagnostics.training import (
    reward_learning_curve,
    exploration_curve,
    q_learning_curve,
    task_performance_summary,
)

from diagnostics.transfer import (
    transfer_matrix,
    transfer_improvement_matrix,
    compare_adaptation_modes,
    compare_search_strategies,
)


# ==========================================================
# Training
# ==========================================================

def plot_reward_learning_curve(logger):
    """
    Reward and rolling reward.
    """
    df = reward_learning_curve(logger)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["episode"],
        df["reward"],
        alpha=0.25,
        label="Reward",
    )

    ax.plot(
        df["episode"],
        df["rolling_mean"],
        linewidth=2,
        label="Rolling mean",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Learning Curve")
    ax.legend()

    return fig


def plot_exploration_curve(logger):
    """
    Exploration diagnostics.
    """
    df = exploration_curve(logger)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    if "action_entropy" in df.columns:
        ax.plot(
            df["episode"],
            df["action_entropy"],
            label="Action entropy",
        )

    if "n_unique_specs" in df.columns:
        ax.plot(
            df["episode"],
            df["n_unique_specs"],
            label="Unique specs",
        )

    ax.set_xlabel("Episode")
    ax.set_title("Exploration Behaviour")
    ax.legend()

    return fig


def plot_loss_curve(logger):
    """
    DQN optimization diagnostics.
    """
    df = q_learning_curve(logger)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["episode"],
        df["loss"],
    )

    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")

    return fig


def plot_q_values(logger):
    """
    Mean Q-values vs targets.
    """
    df = q_learning_curve(logger)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["episode"],
        df["q_mean"],
        label="Q",
    )

    ax.plot(
        df["episode"],
        df["target_q_mean"],
        label="Target Q",
    )

    ax.legend()
    ax.set_title("Q-value Evolution")

    return fig


def plot_task_performance(logger):
    """
    Average reward by task.
    """
    df = task_performance_summary(logger)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["task_name"],
        df["reward_mean"],
    )

    ax.set_ylabel("Reward")
    ax.set_title("Task Performance")

    plt.xticks(rotation=45)

    return fig


# ==========================================================
# Transfer
# ==========================================================

def plot_adaptation_modes(results):
    """
    Compare adaptation modes.
    """
    df = compare_adaptation_modes(results)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        df["adaptation_mode"],
        df["reward_mean"],
    )

    ax.set_ylabel("Reward")
    ax.set_title("Adaptation Modes")

    return fig


def plot_search_strategies(results):
    """
    Compare search strategies.
    """
    df = compare_search_strategies(results)

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        df["search_strategy"],
        df["reward_mean"],
    )

    ax.set_ylabel("Reward")
    ax.set_title("Search Strategies")

    return fig


def plot_transfer_matrix(results):
    """
    Train dataset × test dataset.
    """
    matrix = transfer_matrix(results)

    if matrix.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix.values)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title("Transfer Matrix")

    plt.colorbar(im)

    return fig


def plot_transfer_improvement_matrix(results):
    """
    Adaptation gains.
    """
    matrix = transfer_improvement_matrix(results)

    if matrix.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix.values)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title("Transfer Improvement Matrix")

    plt.colorbar(im)

    return fig


# ==========================================================
# Embeddings
# ==========================================================

def plot_embedding_pca(logger, task_id):
    """
    PCA visualization of specification embeddings.
    """
    embeddings = logger.get_embeddings(task_id)

    if embeddings is None:
        return None

    if len(embeddings) < 3:
        return None

    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        alpha=0.6,
    )

    ax.set_title(f"PCA Embeddings - Task {task_id}")

    return fig


def plot_embedding_umap(logger, task_id):
    """
    UMAP visualization of specification embeddings.
    """
    embeddings = logger.get_embeddings(task_id)

    if embeddings is None:
        return None

    if len(embeddings) < 10:
        return None

    import umap

    coords = umap.UMAP().fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        alpha=0.6,
    )

    ax.set_title(f"UMAP Embeddings - Task {task_id}")

    return fig