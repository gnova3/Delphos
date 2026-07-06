from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.task_registry import TASKS
from training.trainer import MultiTaskTrainer

def generate_linear_epsilon_schedule(n_rounds: int = 5_000, initial_epsilon: float = 1.0, final_epsilon: float = 0.01, num_steps: int = 5_000) -> dict:
    """Generate a linear epsilon decay schedule."""
    epsilon_values = np.linspace(initial_epsilon, final_epsilon, num=num_steps)
    rounds_per_step = n_rounds // num_steps
    schedule = {float(eps): int(rounds_per_step) for eps in epsilon_values}
    return schedule


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_divide(a, b):
    return np.nan if b == 0 else a / b


def analyse_training(trainer: MultiTaskTrainer) -> None:
    print_header("POST-TRAINING DEBUG SUMMARY")

    # 1. REPLAY AND UPDATE COUNTS
    print("1. REPLAY AND UPDATE COUNTS")
    replay_size = len(trainer.replay_buffer)
    print(f"Replay size: {replay_size}")
    print(f"Learn steps: {trainer.learn_steps}")

    if replay_size == 0:
        print("WARNING: Replay buffer is empty.")
    if trainer.learn_steps == 0:
        print("WARNING: No gradient updates happened. Batch size may be too large or replay never became sampleable.")

    # Replay summary if available
    try:
        if hasattr(trainer.replay_buffer, "summary"):
            print("\nReplay summary:")
            print(trainer.replay_buffer.summary())
    except Exception as exc:
        print(f"Replay summary unavailable: {exc}")

    # Build DataFrames
    training_df = pd.DataFrame(trainer.training_log) if trainer.training_log else pd.DataFrame()
    buffer_df = pd.DataFrame(trainer.buffer_log) if trainer.buffer_log else pd.DataFrame()

    print(f"\nTraining log rows: {len(training_df)}")
    print(f"Buffer log rows:   {len(buffer_df)}")

    if training_df.empty:
        print("WARNING: training_log is empty.")
        return

    # Separate task rows from round update rows
    task_training_df = training_df[training_df["task"] != "__round_update__"].copy()
    round_update_df = training_df[training_df["task"] == "__round_update__"].copy()

    # 2. UPDATE QUALITY CHECKS
    print_header("2. UPDATE QUALITY CHECKS")
    if not round_update_df.empty and "loss" in round_update_df.columns:
        losses = pd.to_numeric(round_update_df["loss"], errors="coerce")
        print(f"Round updates logged: {len(round_update_df)}")
        print(f"NaN losses: {losses.isna().sum()}")
        print(f"Unique loss values: {losses.nunique(dropna=True)}")
        print(f"Loss summary:\n{losses.describe()}")
    else:
        print("No round-level loss diagnostics available.")

    # 3. TERMINATION / FORCED STOP DIAGNOSTICS
    print_header("3. TERMINATION / FORCED STOP DIAGNOSTICS")
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

                print("\nMean realized trajectory length by task:")
                print(tmp.groupby("task")["L_e"].mean().sort_values(ascending=False))
    else:
        print("No task-level training rows available.")

    # 4. REWARD IMPROVEMENT BY EPSILON PHASE
    print_header("4. REWARD IMPROVEMENT BY EPSILON PHASE")
    if not task_training_df.empty and {"epsilon", "reward"}.issubset(task_training_df.columns):
        phase_summary = (
            task_training_df.groupby("epsilon")["reward"]
            .agg(["mean", "std", "size"])
            .sort_index(ascending=False)
        )
        print(phase_summary)

        print("\nMean reward by epsilon and task:")
        reward_by_phase_task = (
            task_training_df.groupby(["epsilon", "task"])["reward"]
            .mean()
            .unstack("task")
            .sort_index(ascending=False)
        )
        print(reward_by_phase_task)

    # 5. REPLAY BALANCE ACROSS TASKS
    print_header("5. REPLAY BALANCE ACROSS TASKS")
    if not buffer_df.empty and "task" in buffer_df.columns:
        task_counts = buffer_df.groupby("task").size().sort_values(ascending=False)
        print("Transitions stored by task:")
        print(task_counts)
        if len(task_counts) > 1:
            imbalance_ratio = task_counts.max() / max(1, task_counts.min())
            print(f"\nReplay imbalance ratio: {imbalance_ratio:.3f}")
    else:
        print("No buffer log information available for replay balance.")

    # 6. POLICY COLLAPSE CHECKS
    print_header("6. POLICY COLLAPSE CHECKS")
    if not buffer_df.empty and "action_index" in buffer_df.columns:
        action_freq = buffer_df["action_index"].value_counts(normalize=True).sort_index()
        print("Action frequency distribution:")
        print(action_freq)

        # Repeated patterns
        if {"task", "state_len", "action_index"}.issubset(buffer_df.columns):
            pattern_counts = (
                buffer_df.groupby(["task", "state_len", "action_index"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            print("\nMost frequent (task, state_len, action_index) patterns:")
            print(pattern_counts.head(10).to_string(index=False))
    else:
        print("No buffer data available for behavior analysis.")

    # 7. QUICK TAILS
    print_header("7. QUICK TAILS")
    if not task_training_df.empty:
        print("Task-level training log tail:")
        print(task_training_df.tail(10).to_string(index=False))


def main() -> None:
    # Use a small schedule for debugging
    debug_epsilon_schedule = {
        1.00: 10,
        0.50: 10,
        0.10: 10,
    }

    trainer = MultiTaskTrainer(
        tasks=TASKS,
        encoder_kind="deepset",
        linear_additive=False,
        epsilon_schedule=debug_epsilon_schedule,
        batch_size=32,
        buffer_size=10_000,
        replay_kind="uniform",
        learning_rate=1e-3,
        discount_factor=0.95,
        target_soft_tau=5e-3,
        device="cpu",
        seed=123,
        batch_log_size=10,
        enable_file_logging=True,
        enable_parquet_logs=False,
        enable_checkpoints=True,
        save_dir="experiments/debug",
        horizon_kappa=1.5,
        step_penalty_lambda=0.01,
    )

    print_header("RUN CONFIGURATION")
    print(f"Number of tasks: {len(TASKS)}")
    print(f"Tasks: {[task.name for task in TASKS]}")
    print(f"Context dim: {trainer.agent.context_dim}")
    print(f"Num actions: {trainer.agent.num_actions}")
    print(f"Epsilon schedule: {debug_epsilon_schedule}")

    print("\nPer-task horizons:")
    for task in TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        H_tau = trainer.get_task_horizon(runtime)
        print(f"  - {task.name:20s}: H_tau={H_tau}")

    print("\nStarting debug training run...")
    trainer.train()
    
    analyse_training(trainer)


if __name__ == "__main__":
    main()