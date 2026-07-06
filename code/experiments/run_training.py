from __future__ import annotations

import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.task_registry import TASKS
from training.trainer import MultiTaskTrainer


def build_linear_epsilon_schedule(n_steps: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05, rounding: int = 5,) -> OrderedDict[float, int]:
    if n_steps == 1:
        return OrderedDict({round(epsilon_start, rounding): 1})

    decay_rate = (epsilon_start - epsilon_end) / (n_steps - 1)

    schedule = OrderedDict()
    for step in range(n_steps):
        epsilon = epsilon_start - decay_rate * step
        epsilon = max(epsilon_end, epsilon)
        epsilon = round(epsilon, rounding)
        schedule[epsilon] = schedule.get(epsilon, 0) + 1

    return schedule


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def analyse_training(trainer: MultiTaskTrainer) -> None:
    print_header("POST-TRAINING SUMMARY")

    # 1. REPLAY AND UPDATE COUNTS
    print("1. REPLAY AND UPDATE COUNTS")
    replay_size = len(trainer.replay_buffer)
    print(f"Replay size: {replay_size}")
    print(f"Learn steps: {trainer.learn_steps}")

    if replay_size == 0:
        print("WARNING: Replay buffer is empty.")
    if trainer.learn_steps == 0:
        print("WARNING: No gradient updates happened.")

    # Build DataFrames
    training_df = pd.DataFrame(trainer.training_log) if trainer.training_log else pd.DataFrame()
    buffer_df = pd.DataFrame(trainer.buffer_log) if trainer.buffer_log else pd.DataFrame()

    if training_df.empty:
        print("WARNING: training_log is empty.")
        return

    # Separate task rows from round update rows
    task_training_df = training_df[training_df["task"] != "__round_update__"].copy()

    # 2. TERMINATION / FORCED STOP DIAGNOSTICS
    print_header("2. TERMINATION / FORCED STOP DIAGNOSTICS")
    if not task_training_df.empty:
        if "forced_stop" in task_training_df.columns:
            print("Mean forced_stop by task:")
            print(task_training_df.groupby("task")["forced_stop"].mean().sort_values(ascending=False))

        if {"task", "L_e", "H_tau"}.issubset(task_training_df.columns):
            tmp = task_training_df.dropna(subset=["L_e", "H_tau"]).copy()
            if not tmp.empty:
                tmp["L_over_H"] = tmp["L_e"] / tmp["H_tau"]
                print("\nMean L_e / H_tau by task:")
                print(tmp.groupby("task")["L_over_H"].mean().sort_values(ascending=False))
    else:
        print("No task-level training rows available.")

    # 3. REWARD IMPROVEMENT BY EPSILON PHASE
    print_header("3. REWARD IMPROVEMENT BY EPSILON PHASE")
    if not task_training_df.empty and {"epsilon", "reward"}.issubset(task_training_df.columns):
        # We take a sample of unique epsilons to avoid too much noise if there are many
        unique_eps = sorted(task_training_df["epsilon"].unique(), reverse=True)
        sample_eps = [unique_eps[0], unique_eps[len(unique_eps)//2], unique_eps[-1]]
        
        phase_summary = (
            task_training_df[task_training_df["epsilon"].isin(sample_eps)]
            .groupby("epsilon")["reward"]
            .agg(["mean", "size"])
            .sort_index(ascending=False)
        )
        print("Reward sample across epsilon phases:")
        print(phase_summary)

    # 4. REPLAY BALANCE ACROSS TASKS
    print_header("4. REPLAY BALANCE ACROSS TASKS")
    if not buffer_df.empty and "task" in buffer_df.columns:
        task_counts = buffer_df.groupby("task").size().sort_values(ascending=False)
        print("Transitions stored by task:")
        print(task_counts)


def main() -> None:
    n_episodes = 10_000
    epsilon_schedule = build_linear_epsilon_schedule(n_episodes)
    run_name = f"mt_deepset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path("experiments") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "n_episodes": n_episodes,
        "n_tasks": len(TASKS),
        "encoder_kind": "deepset",
        "linear_additive": False,
        "batch_size": 64,
        "buffer_size": 40_000,
        "replay_kind": "uniform",
        "replay_kwargs": {},
        "learning_rate": 1e-3,
        "discount_factor": 0.90,
        "target_soft_tau": 5e-3,
        "grad_clip_norm": 1.0,
        "head_flag": True,
        "pooling": "mean",
        "device": "cpu",
        "use_amp": False,
        "seed": 123,
        "batch_log_size": 50,
        "enable_file_logging": True,
        "enable_parquet_logs": False,
        "enable_csv_fallback": True,
        "enable_checkpoints": True,
        "save_final_checkpoint": True,
        "checkpoint_every": 1000,
        "save_dir": str(save_dir),
        "resume_path": None,
        "epsilon_schedule_unique_steps": len(epsilon_schedule),
        "epsilon_schedule_total_steps": int(sum(epsilon_schedule.values())),
        "horizon_kappa": 1.5,
        "step_penalty_lambda": 0.0,
    }

    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print_header("STARTING PRODUCTION TRAINING")
    print(f"Tasks: {[t.name for t in TASKS]}")
    print(f"Save dir: {save_dir}")
    print(f"Epsilon schedule steps: {sum(epsilon_schedule.values())}")

    trainer = MultiTaskTrainer(
        tasks=TASKS,
        encoder_kind="deepset",
        linear_additive=False,
        epsilon_schedule=epsilon_schedule,
        batch_size=64,
        buffer_size=40_000,
        replay_kind="uniform",
        replay_kwargs={},
        learning_rate=1e-3,
        discount_factor=0.90,
        target_soft_tau=5e-3,
        grad_clip_norm=1.0,
        head_flag=True,
        pooling="mean",
        device="cpu",
        use_amp=False,
        seed=123,
        batch_log_size=50,
        enable_file_logging=True,
        enable_parquet_logs=False,
        enable_csv_fallback=True,
        enable_checkpoints=True,
        save_final_checkpoint=True,
        checkpoint_every=1000,
        save_dir=str(save_dir),
        resume_path=None,
        horizon_kappa=1.5,
        step_penalty_lambda=0.0,
    )
    
    try:
        trainer.train()
    finally:
        analyse_training(trainer)

    print("\nTraining finished.")
    print(f"Artifacts saved under: {save_dir}")


if __name__ == "__main__":
    main()