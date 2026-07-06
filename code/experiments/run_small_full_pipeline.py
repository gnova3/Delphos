# =============================================================================
# DELPHOS END-TO-END VALIDATION PIPELINE
# =============================================================================
#
# Purpose
# -------
# Full-system validation for Delphos multitask RL framework.
#
# This script validates:
#
# PHASE 1  - Infrastructure / runtime correctness
# PHASE 2  - RL stability and learning updates
# PHASE 3  - Training diagnostics and behaviour
# PHASE 4  - Checkpoint reproducibility
# PHASE 5  - Frozen-agent transfer to unseen dataset
# PHASE 6  - Final scientific summaries and diagnostics
#
# IMPORTANT
# ---------
# - Frozen inference on unseen task
# - Extensive diagnostics and failure checks
#
# =============================================================================
from __future__ import annotations
from pandas._typing import F

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# =============================================================================
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# =============================================================================
# Imports
# =============================================================================

from configs.task_registry import TASKS, INFERENCE_TASKS
from training.trainer import MultiTaskTrainer
from mdp.task_runtime import evaluate_final_terms, initial_terms_for_task
from utils.checkpointing import load_agent_checkpoint
from utils.logging_utils import get_delphos_logger
from env.result_cache import load_all_rewards
from diagnostics.diagnostics_logger import DiagnosticsLogger
from diagnostics.diagnostics_summary import export_all_csvs, compute_per_task_summary, summarize_training
from transfer.transfer_evaluation import compare_inference_modes, summarize_transfer_df

LOGGER = get_delphos_logger("Delphos.small_full_pipeline")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Train and inference tasks
TRAIN_TASKS = TASKS
INFERENCE_TASKS = [INFERENCE_TASKS[0]]

# Training parameters
EPSILON_SCHEDULE = {1.0: 200,
                    0.98: 200,
                    0.96: 200,
                    0.94: 200,
                    0.92: 200,
                    0.90: 200,
                    0.88: 200,
                    0.86: 200,
                    0.84: 200,
                    0.82: 200,
                    0.80: 200,
                    0.78: 200,
                    0.76: 200,
                    0.74: 200,
                    0.72: 200,
                    0.70: 200,
                    0.68: 200,
                    0.66: 200,
                    0.64: 200,
                    0.62: 200,
                    0.60: 200,
                    0.58: 200,
                    0.56: 200,
                    0.54: 200,
                    0.52: 200,
                    0.50: 200,
                    0.48: 200,
                    0.46: 200,
                    0.44: 200,
                    0.42: 200,
                    0.40: 200,
                    0.38: 200,
                    0.36: 200,
                    0.34: 200,
                    0.32: 200,
                    0.30: 200,
                    0.28: 200,
                    0.26: 200,
                    0.24: 200,
                    0.22: 200,
                    0.20: 200,
                    0.18: 200,
                    0.16: 200,
                    0.14: 200,
                    0.12: 200,
                    0.10: 200,
                    0.08: 200,
                    0.06: 200,
                    0.04: 200,
                    0.02: 200,
                    0.00: 200
                    }
                    
LINEAR_ADDITIVE = True
BATCH_SIZE = 8
BUFFER_SIZE = 256
SEED = 123
DEVICE = "cpu"

# Transfer evaluation
N_TRANSFER_EPISODES = 100

# Output
RUN_ROOT = REPO_ROOT / "experiments" / "small_full_pipeline"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def terms_to_string(terms) -> str:
    return " | ".join(f"({t.attribute_id},{t.transformation_id},{t.taste_id},{t.covariate_id})" for t in terms)


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def has_nan_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item())


def has_inf_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isinf(x).any().item())


# =============================================================================
# SQLITE
# =============================================================================
def inspect_rewards_sqlite(task) -> dict:
    rewards_dir = REPO_ROOT / Path(task.path_rewards)
    db_path = rewards_dir / "rewards.sqlite"

    if not db_path.exists():
        return {"task_name": task.df_name, "db_exists": False, "n_rows": 0,}

    df = load_all_rewards(db_path)

    return {
        "task_name": task.df_name,
        "db_exists": True,
        "n_rows": int(len(df)),
        "n_success": int(
            df["successfulEstimation"]
            .fillna(0)
            .astype(int)
            .sum()
        )
        if "successfulEstimation" in df.columns
        else None,
    }


# =============================================================================
# CHECKPOINTS
# =============================================================================
def get_latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None

    candidates = sorted(run_dir.glob("checkpoint_*.pt"))

    if candidates:
        return candidates[-1]

    final_ckpt = run_dir / "delphos_agent.pt"
    if final_ckpt.exists():
        return final_ckpt

    return None


# =============================================================================
# ENVIRONMENT WRAPPER
# =============================================================================
def trainer_get_mnl_outcomes_with_debug(*args, **kwargs):
    try:
        return get_mnl_outcomes_debug_passthrough(*args, **kwargs)
    except Exception as exc:
        LOGGER.exception("get_mnl_outcomes failed during validation.")
        raise exc

def get_mnl_outcomes_debug_passthrough(*args, **kwargs):
    from env.environment import get_mnl_outcomes
    kwargs = dict(kwargs)
    kwargs.setdefault("debug_apollo", False)
    kwargs.setdefault("info", False)
    kwargs.setdefault("save", False)
    return get_mnl_outcomes(*args, **kwargs)


# =============================================================================
# PHASE 1
# Infrastructure validation
# =============================================================================

def phase_1_validate_runtimes(trainer):
    print("\n" + "=" * 100)
    print("PHASE 1 - INFRASTRUCTURE VALIDATION")
    print("=" * 100)

    assert_true(len(trainer.runtimes_by_task) == len(TRAIN_TASKS),"Mismatch in runtimes_by_task.")
    rows = []

    for task in TRAIN_TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        am = runtime.action_manager
        assert_true(runtime.task.task_id == task.task_id, f"Task mismatch for {task.name}")
        assert_true(am.num_catalogue_actions == trainer.agent.num_actions, f"Action-space mismatch for {task.name}")

        rows.append(
            {
                "task_name": task.name,
                "n_attributes": len(runtime.state_manager.task.attribute_ids),
                "n_covariates": len(runtime.state_manager.task.covariate_ids),
                "n_actions": am.num_catalogue_actions,
                "linear_additive": am.linear_additive,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# =============================================================================
# PHASE 2
# RL stability diagnostics
# =============================================================================

def phase_2_pretrain_rollouts(trainer):

    print("\n" + "=" * 100)
    print("PHASE 2 - PRETRAIN ROLLOUT VALIDATION")
    print("=" * 100)

    rows = []

    for task in TRAIN_TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        (steps, final_terms, L_e, H_tau, forced_stop,) = trainer.run_episode(runtime)

        spec_key, reward = evaluate_final_terms(
            runtime=runtime,
            final_terms=final_terms,
            get_mnl_fn=trainer.get_mnl_fn,
            return_spec=True,
            suppress_exceptions=False,
            logger=LOGGER,
        )

        # -----------------------------------------------------------------
        # Mask diagnostics
        # -----------------------------------------------------------------
        current_terms = initial_terms_for_task(runtime, trainer.linear_additive)
        visited_state_keys = {runtime.action_manager.canonical_state_key(current_terms)}

        masks = runtime.action_manager.get_action_masks(current_state=current_terms,visited_state_keys=visited_state_keys,)

        rows.append(
            {
                "task_name": task.name,
                "reward": safe_float(reward),
                "L_e": int(L_e),
                "H_tau": int(H_tau),
                "forced_stop": bool(forced_stop),
                "n_terms": int(len(final_terms)),
                "task_mask_count": int(masks["task_mask"].sum().item()),
                "state_mask_count": int(masks["state_action_mask"].sum().item()),
                "repeated_mask_count": int(masks["repeated_action_mask"].sum().item()),
                "trajectory_mask_count": int(masks["trajectory_mask"].sum().item()),
                "valid_mask_count": int(masks["valid_mask"].sum().item()),
                "specification": spec_key,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# =============================================================================
# PHASE 3
# Training + diagnostics
# =============================================================================

def phase_3_train(trainer):

    print("\n" + "=" * 100)
    print("PHASE 3 - TRAINING")
    print("=" * 100)

    # Attach diagnostics logger before training
    diag_logger = DiagnosticsLogger(cap=200_000)
    trainer.diag_logger = diag_logger

    policy_before = {k: v.detach().clone() for k, v in trainer.agent.policy_net.state_dict().items()}
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    assert_true(trainer.learn_steps > 0, "No learning steps happened.")
    assert_true(len(trainer.replay_buffer) > 0, "Replay buffer is empty.")

    policy_after = trainer.agent.policy_net.state_dict()
    policy_changed = any(not torch.equal(policy_before[k], policy_after[k]) for k in policy_before)
    assert_true(policy_changed, "Policy network did not change.")

    # Export diagnostics CSVs
    run_dir = Path(trainer.subfolder)
    task_name_map = {task.task_id: task.name for task in TRAIN_TASKS}
    diag_summary = summarize_training(
        logger=diag_logger,
        run_dir=run_dir,
        task_name_map=task_name_map,
        buffer=trainer.replay_buffer,
    )
    LOGGER.info("Diagnostics exported to %s", run_dir)

    # ---------------------------------------------------------------------
    # Q diagnostics
    # ---------------------------------------------------------------------

    q_stats = []

    for task in TRAIN_TASKS:

        runtime = trainer.runtimes_by_task[task.task_id]
        initial_terms = initial_terms_for_task(runtime, trainer.linear_additive)

        z = trainer.agent.encode_terms(initial_terms, runtime)

        q = (
            trainer.agent.policy_net(z.unsqueeze(0))
            .detach()
            .cpu()
            .flatten()
        )
        assert_true(not has_nan_tensor(q), f"NaN Q-values for {task.name}")
        assert_true(not has_inf_tensor(q), f"Inf Q-values for {task.name}")

        q_stats.append(
            {
                "task_name": task.name,
                "q_min": float(q.min().item()),
                "q_max": float(q.max().item()),
                "q_mean": float(q.mean().item()),
                "q_std": float(q.std().item()),
            }
        )

    q_df = pd.DataFrame(q_stats)

    print(q_df.to_string(index=False))

    return {
        "elapsed_seconds": elapsed,
        "policy_changed": policy_changed,
        "q_df": q_df,
        "diag_logger": diag_logger,
        "diag_summary": diag_summary,
    }


# =============================================================================
# PHASE 4
# Checkpoint reproducibility
# =============================================================================


def phase_4_checkpoint_validation(trainer):

    print("\n" + "=" * 100)
    print("PHASE 4 - CHECKPOINT VALIDATION")
    print("=" * 100)

    run_dir = Path(trainer.subfolder)

    final_ckpt = run_dir / "delphos_agent.pt"

    assert_true(
        final_ckpt.exists(),
        "Final checkpoint missing.",
    )

    # ---------------------------------------------------------------------
    # Reload trainer
    # ---------------------------------------------------------------------

    reloaded_trainer = MultiTaskTrainer(
        tasks=TRAIN_TASKS,
        encoder_kind="deepset",
        linear_additive=LINEAR_ADDITIVE,
        epsilon_schedule={1.0: 1},
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        replay_kind="uniform",
        replay_kwargs={},
        learning_rate=1e-3,
        discount_factor=0.90,
        target_soft_tau=5e-3,
        grad_clip_norm=1.0,
        head_flag=True,
        pooling="mean",
        device=DEVICE,
        use_amp=False,
        seed=SEED,
        enable_checkpoints=False,
        save_final_checkpoint=False,
        get_mnl_fn=trainer.get_mnl_fn,
    )

    load_agent_checkpoint(
        str(final_ckpt),
        agent=reloaded_trainer.agent,
        optimizer=reloaded_trainer.agent.optimizer,
        mode="resume",
        strict=False,
        load_target=True,
        reset_output_if_mismatch=True,
    )

    # ---------------------------------------------------------------------
    # Compare outputs
    # ---------------------------------------------------------------------

    sample_task = TRAIN_TASKS[0]

    runtime_orig = trainer.runtimes_by_task[sample_task.task_id]
    runtime_new = reloaded_trainer.runtimes_by_task[sample_task.task_id]

    z_orig = trainer.agent.encode_terms([], runtime_orig)

    z_new = reloaded_trainer.agent.encode_terms([], runtime_new)

    q_orig = (
        trainer.agent.policy_net(z_orig.unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )

    q_new = (
        reloaded_trainer.agent.policy_net(z_new.unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )

    assert_true(np.allclose(q_orig, q_new, atol=1e-6), "Reloaded Q-values mismatch.")
    print("Checkpoint reload validated.")
    return reloaded_trainer


# =============================================================================
# PHASE 5
# Frozen-agent transfer
# =============================================================================

def phase_5_transfer(reloaded_trainer):

    print("\n" + "=" * 100)
    print("PHASE 5 - FROZEN AGENT TRANSFER")
    print("=" * 100)

    reloaded_trainer.inference()
    unseen_runtime = reloaded_trainer.register_runtime(INFERENCE_TASKS[0])

    # Inference diagnostics logger for transfer phase
    inf_logger = DiagnosticsLogger(cap=50_000)

    # Compare three inference modes
    transfer_df = compare_inference_modes(
        trainer=reloaded_trainer,
        runtime=unseen_runtime,
        modes=["greedy", "stochastic", "boltzmann"],
        n_episodes=N_TRANSFER_EPISODES,
        epsilon=0.05,
        temperature=0.5,
        get_mnl_fn=reloaded_trainer.get_mnl_fn,
        diag_logger=inf_logger,
    )

    print(transfer_df.to_string(index=False))

    mode_summary_df = summarize_transfer_df(transfer_df)
    print("\n--- Mode Summary ---")
    print(mode_summary_df.to_string(index=False))

    transfer_summary = {
        "mean_reward":   float(transfer_df["reward"].mean()) if not transfer_df.empty else float("nan"),
        "std_reward":    float(transfer_df["reward"].std())  if not transfer_df.empty else float("nan"),
        "unique_specs":  int(transfer_df["spec_key"].nunique()) if not transfer_df.empty else 0,
        "mean_terms":    float(transfer_df["n_terms"].mean()) if not transfer_df.empty else float("nan"),
        "mean_steps":    float(transfer_df["n_steps"].mean()) if not transfer_df.empty else float("nan"),
        "modes_tested":  ["greedy", "stochastic", "boltzmann"],
    }

    print(json.dumps(transfer_summary, indent=2))

    return transfer_df, transfer_summary, inf_logger


# =============================================================================
# PHASE 6
# Final summaries
# =============================================================================


def phase_6_final_summary(
    trainer,
    train_rollout_df,
    train_stats,
    transfer_df,
    transfer_summary,
    inf_logger=None,
):

    print("\n" + "=" * 100)
    print("PHASE 6 - FINAL SUMMARY")
    print("=" * 100)

    run_dir = Path(trainer.subfolder)

    # SQLite diagnostics
    sqlite_rows = []
    for task in TRAIN_TASKS + [INFERENCE_TASKS[0]]:
        sqlite_rows.append(inspect_rewards_sqlite(task))
    sqlite_df = pd.DataFrame(sqlite_rows)

    # Save outputs
    train_rollout_df.to_csv(run_dir / "phase2_rollouts.csv", index=False)
    train_stats["q_df"].to_csv(run_dir / "phase3_q_stats.csv", index=False)
    transfer_df.to_csv(run_dir / "phase5_transfer.csv", index=False)
    sqlite_df.to_csv(run_dir / "sqlite_summary.csv", index=False)

    # Export inference diagnostics if available
    if inf_logger is not None:
        inf_df = inf_logger.inference_df
        if not inf_df.empty:
            inf_df.to_csv(run_dir / "inference_diagnostics.csv", index=False)

    # Final manifest
    summary = {
        "training_tasks": [t.name for t in TRAIN_TASKS],
        "unseen_task": INFERENCE_TASKS[0].name,
        "episodes": int(trainer.episode_count),
        "learn_steps": int(trainer.learn_steps),
        "replay_size": int(len(trainer.replay_buffer)),
        "policy_changed": bool(train_stats["policy_changed"]),
        "transfer_summary": transfer_summary,
        "elapsed_seconds": float(train_stats["elapsed_seconds"]),
        "run_dir": str(run_dir),
        "diag_summary": train_stats.get("diag_summary", {}),
    }

    with open(run_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(
        {k: v for k, v in summary.items() if k != "diag_summary"},
        indent=2
    ))


# =============================================================================
# MAIN
# =============================================================================


def main():

    set_seed(SEED)

    print("=" * 100)
    print("DELPHOS END-TO-END VALIDATION")
    print("=" * 100)

    # =========================================================================
    # Build trainer
    # =========================================================================

    trainer = MultiTaskTrainer(
        tasks=TRAIN_TASKS,
        encoder_kind="deepset",
        linear_additive=LINEAR_ADDITIVE,
        epsilon_schedule=EPSILON_SCHEDULE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        replay_kind="uniform",
        replay_kwargs={},
        learning_rate=1e-3,
        discount_factor=0.90,
        target_soft_tau=5e-3,
        grad_clip_norm=1.0,
        head_flag=True,
        pooling="mean",
        device=DEVICE,
        use_amp=False,
        seed=SEED,
        batch_log_size=64,
        enable_file_logging=True,
        enable_parquet_logs=False,
        enable_csv_fallback=True,
        enable_checkpoints=True,
        save_final_checkpoint=True,
        checkpoint_every=500,
        save_dir=str(RUN_ROOT),
        resume_path=None,
        get_mnl_fn=trainer_get_mnl_outcomes_with_debug,
        horizon_kappa=1.5,
        step_penalty_lambda=0.0,
    )

    # =========================================================================
    # PHASE 1
    # =========================================================================

    phase_1_validate_runtimes(trainer)

    # =========================================================================
    # PHASE 2
    # =========================================================================

    train_rollout_df = phase_2_pretrain_rollouts(
        trainer
    )

    # =========================================================================
    # PHASE 3
    # =========================================================================

    train_stats = phase_3_train(trainer)

    # =========================================================================
    # PHASE 4
    # =========================================================================

    reloaded_trainer = phase_4_checkpoint_validation(
        trainer
    )

    # =========================================================================
    # PHASE 5
    # =========================================================================

    transfer_df, transfer_summary, inf_logger = (
        phase_5_transfer(reloaded_trainer)
    )

    # =========================================================================
    # PHASE 6
    # =========================================================================

    phase_6_final_summary(
        trainer=trainer,
        train_rollout_df=train_rollout_df,
        train_stats=train_stats,
        transfer_df=transfer_df,
        transfer_summary=transfer_summary,
        inf_logger=inf_logger,
    )

    print("\n")
    print("=" * 100)
    print("END-TO-END VALIDATION PASSED")
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()