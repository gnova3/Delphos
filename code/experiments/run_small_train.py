from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from configs.task_registry import TASKS
from training.trainer import MultiTaskTrainer
from mdp.task_runtime import evaluate_final_terms, initial_terms_for_task
from utils.checkpointing import load_agent_checkpoint
from utils.logging_utils import get_delphos_logger
from env.result_cache import load_all_rewards


LOGGER = get_delphos_logger("Delphos.small_train")


# =============================================================================
# Configuration
# =============================================================================
DECISIONS_TASK = TASKS[1] # 1: SwissmetroRouteChoice
AUX_TASK = TASKS[0] # ApolloModeChoice
SMALL_TASKS = TASKS
N_DECISIONS_PROPOSALS = 10
DECISIONS_PROPOSAL_TEMPERATURE = 0.5
DECISIONS_PROPOSAL_MAX_ATTEMPTS = 100

N_ROUNDS = 4
EPSILON_SCHEDULE = {1.0: 200, 0.5: 500, 0.4: 500, 0.3: 500, 0.2: 5000, 0.1: 5000, 0.05: 5000, 0.01: 5000}
EPSILON_SCHEDULE = {1.0: 20, 0.01: 20}
BATCH_SIZE = 8
BUFFER_SIZE = 128
SEED = 123
DEVICE = "cpu"

RUN_ROOT = REPO_ROOT / "experiments" / "small_train"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
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


def inspect_rewards_sqlite(task) -> dict:
    rewards_dir = REPO_ROOT / Path(task.path_rewards)
    db_path = rewards_dir / "rewards.sqlite"
    if not db_path.exists():
        return {
            "task_name": task.df_name,
            "db_exists": False,
            "n_rows": 0,
        }

    df = load_all_rewards(db_path)
    return {
        "task_name": task.df_name,
        "db_exists": True,
        "n_rows": int(len(df)),
        "n_success": int(df["successfulEstimation"].fillna(0).astype(int).sum()) if "successfulEstimation" in df.columns else None,
    }


def compare_weights(model_a: torch.nn.Module, model_b: torch.nn.Module) -> dict:
    diffs = {}
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            diffs[k] = float(torch.max(torch.abs(sd_a[k] - sd_b[k])).item())
    return diffs


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
# Helper functions for Decisions proposals
# =============================================================================

def terms_to_string(terms) -> str:
    return " | ".join(
        f"({t.attribute_id},{t.transformation_id},{t.taste_id},{t.covariate_id})"
        for t in terms
    )


def generate_decisions_proposals(
    trainer: MultiTaskTrainer,
    runtime,
    n_proposals: int,
    temperature: float,
    max_attempts: int,
):
    proposals = []
    seen_specs = set()
    attempts = 0

    while len(proposals) < n_proposals and attempts < max_attempts:
        attempts += 1
        steps, final_terms, L_e, H_tau, forced_stop = trainer.run_episode(
            runtime,
            boltzmann=True,
            temperature=temperature,
        )

        spec_key, reward = evaluate_final_terms(
            runtime=runtime,
            final_terms=final_terms,
            get_mnl_fn=trainer.get_mnl_fn,
            return_spec=True,
            suppress_exceptions=False,
            logger=LOGGER,
        )

        if not spec_key or spec_key in seen_specs:
            continue

        seen_specs.add(spec_key)
        proposals.append(
            {
                "proposal_id": len(proposals) + 1,
                "task_name": runtime.task.name,
                "specification": spec_key,
                "reward": float(reward),
                "n_steps": int(L_e),
                "H_tau": int(H_tau),
                "forced_stop": bool(forced_stop),
                "n_terms": int(len(final_terms)),
                "terms": terms_to_string(final_terms),
            }
        )

    return pd.DataFrame(proposals), attempts


# =============================================================================
# Main small routine
# =============================================================================
def main() -> None:
    set_seed(SEED)

    started = time.time()
    print("=" * 100)
    print("DELPHOS SMALL TRAIN")
    print("=" * 100)
    print(f"Tasks: {[t.name for t in SMALL_TASKS]}")
    print(f"Epsilon schedule: {EPSILON_SCHEDULE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"Device: {DEVICE}")

    trainer = MultiTaskTrainer(
        tasks=SMALL_TASKS,
        encoder_kind="deepset",
        linear_additive=True,
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
        batch_log_size=3,
        enable_file_logging=True,
        enable_parquet_logs=False,
        enable_csv_fallback=True,
        enable_checkpoints=True,
        save_final_checkpoint=True,
        checkpoint_every=3,
        save_dir=str(RUN_ROOT),
        resume_path=None,
        get_mnl_fn=trainer_get_mnl_outcomes_with_debug,
        horizon_kappa=1.5,
        step_penalty_lambda=0.0,
    )

    print("\n" + "-" * 100)
    print("1. PRE-TRAIN CHECKS")
    print("-" * 100)

    # 1. Check task runtimes and action spaces
    assert_true(len(trainer.runtimes_by_task) == len(SMALL_TASKS), "Mismatch in runtimes_by_task size.")
    print("Runtimes built successfully.")

    for task in SMALL_TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        assert_true(runtime.task.task_id == task.task_id, f"Runtime task id mismatch for {task.name}.")
        assert_true(len(runtime.state_manager.task.attribute_ids) > 0, f"No attribute_ids for {task.name}.")
        assert_true(runtime.action_manager.num_catalogue_actions == trainer.agent.num_actions, f"Action-space mismatch for {task.name}.")
        print(
            f"[{task.name}] "
            f"task_attrs={runtime.state_manager.task.attribute_ids} | "
            f"covs={runtime.state_manager.task.covariate_ids} | "
            f"actions={runtime.action_manager.num_catalogue_actions}"
        )

    # 2. Run one pre-train episode per task
    print("\n" + "-" * 100)
    print("2. PRE-TRAIN EPISODE CHECKS")
    print("-" * 100)

    pretrain_specs = []
    for task in SMALL_TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        steps, final_terms, L_e, H_tau, forced_stop = trainer.run_episode(runtime)

        assert_true(L_e >= 0, f"Invalid episode length for {task.name}.")
        assert_true(H_tau >= 1, f"Invalid horizon for {task.name}.")
        assert_true(isinstance(final_terms, list), f"final_terms is not a list for {task.name}.")

        spec_key, reward = evaluate_final_terms(
            runtime=runtime,
            final_terms=final_terms,
            get_mnl_fn=trainer.get_mnl_fn,
            return_spec=True,
            suppress_exceptions=False,
            logger=LOGGER,            
        )

        assert_true(isinstance(spec_key, str) and len(spec_key) > 0, f"Empty spec_key for {task.name}.")
        assert_true(isinstance(reward, float), f"Reward is not float for {task.name}.")

        pretrain_specs.append((task.name, spec_key))

        print(
            f"[{task.name}] "
            f"L_e={L_e} | H_tau={H_tau} | forced_stop={forced_stop} | "
            f"final_terms={len(final_terms)} | reward={reward:.4f} | spec={spec_key}"
        )

    # 3. Re-evaluate same specs to test SQLite reuse
    print("\n" + "-" * 100)
    print("3. SQLITE CACHE REUSE CHECK")
    print("-" * 100)

    for task_name, spec_key in pretrain_specs:
        task = next(t for t in SMALL_TASKS if t.name == task_name)
        runtime = trainer.runtimes_by_task[task.task_id]

        t0 = time.time()
        out_1 = trainer.get_mnl_fn(
            spec_key,
            [],
            [],
            [],
            runtime.state_manager,
            task.path_rewards,
            task.path_choice_dataset,
            info=False,
            save=False,
            debug_apollo=False,
            raise_on_error=True,
        )
        dt1 = time.time() - t0

        t1 = time.time()
        out_2 = trainer.get_mnl_fn(
            spec_key,
            [],
            [],
            [],
            runtime.state_manager,
            task.path_rewards,
            task.path_choice_dataset,
            info=False,
            save=False,
            debug_apollo=False,
            raise_on_error=True,
        )
        dt2 = time.time() - t1

        assert_true(len(out_1) >= 1, f"First cache lookup failed for {task_name}.")
        assert_true(len(out_2) >= 1, f"Second cache lookup failed for {task_name}.")
        print(f"[{task_name}] first_call={dt1:.3f}s | second_call={dt2:.3f}s")

    # 4. Train
    print("\n" + "-" * 100)
    print("4. TRAINING")
    print("-" * 100)

    policy_before = {k: v.detach().clone() for k, v in trainer.agent.policy_net.state_dict().items()}
    encoder_before = {k: v.detach().clone() for k, v in trainer.agent.encoder.state_dict().items()}

    trainer.train()

    assert_true(trainer.episode_count > 0, "No episodes were counted.")
    assert_true(len(trainer.replay_buffer) > 0, "Replay buffer stayed empty.")
    assert_true(trainer.learn_steps > 0, "No learning updates happened.")

    print(f"Episode count: {trainer.episode_count}")
    print(f"Replay size: {len(trainer.replay_buffer)}")
    print(f"Learn steps: {trainer.learn_steps}")

    # 5. Weight changes
    print("\n" + "-" * 100)
    print("5. PARAMETER CHANGE CHECK")
    print("-" * 100)

    policy_after = trainer.agent.policy_net.state_dict()
    encoder_after = trainer.agent.encoder.state_dict()

    policy_changed = any(
        not torch.equal(policy_before[k], policy_after[k]) for k in policy_before
    )
    encoder_changed = any(
        not torch.equal(encoder_before[k], encoder_after[k]) for k in encoder_before
    )

    assert_true(policy_changed, "Policy network weights did not change.")
    print(f"Policy changed: {policy_changed}")
    print(f"Encoder changed: {encoder_changed}")

    # 6. Experiment artifacts
    print("\n" + "-" * 100)
    print("6. ARTIFACT CHECK")
    print("-" * 100)

    assert_true(trainer.subfolder is not None, "Trainer subfolder was not created.")
    run_dir = Path(trainer.subfolder)
    assert_true(run_dir.exists(), "Run directory does not exist.")

    expected_files = [
        run_dir / "trainer_metadata.json",
        run_dir / "training_log.csv",
        run_dir / "buffer_log.csv",
        run_dir / "delphos_agent.pt",
    ]

    for fp in expected_files:
        assert_true(fp.exists(), f"Expected artifact missing: {fp}")
        print(f"Found artifact: {fp.name}")

    latest_ckpt = get_latest_checkpoint(run_dir)
    assert_true(latest_ckpt is not None and latest_ckpt.exists(), "No checkpoint found.")
    print(f"Latest checkpoint: {latest_ckpt.name}")

    # 7. Reload checkpoint and compare outputs
    print("\n" + "-" * 100)
    print("7. CHECKPOINT RELOAD CHECK")
    print("-" * 100)

    reloaded_trainer = MultiTaskTrainer(
        tasks=SMALL_TASKS,
        encoder_kind="deepset",
        linear_additive=True,
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
        batch_log_size=3,
        enable_file_logging=False,
        enable_parquet_logs=False,
        enable_csv_fallback=False,
        enable_checkpoints=False,
        save_final_checkpoint=False,
        save_dir=None,
        resume_path=None,
        get_mnl_fn=trainer.get_mnl_fn,
    )

    final_ckpt = run_dir / "delphos_agent.pt"
    assert_true(final_ckpt.exists(), "Final checkpoint not found.")
    print(f"Final checkpoint: {final_ckpt.name}")
    
    meta = load_agent_checkpoint(
    str(final_ckpt),
    agent=reloaded_trainer.agent,
    optimizer=reloaded_trainer.agent.optimizer,
    mode="resume",
    strict=False,
    load_target=True,
    reset_output_if_mismatch=True,
)

    print("Reload meta:", meta)

    sample_task = DECISIONS_TASK
    runtime_orig = trainer.runtimes_by_task[sample_task.task_id]
    runtime_new = reloaded_trainer.runtimes_by_task[sample_task.task_id]

    current_terms = []
    z_orig = trainer.agent.encode_terms(current_terms, runtime_orig)
    z_new = reloaded_trainer.agent.encode_terms(current_terms, runtime_new)

    q_orig = trainer.agent.policy_net(z_orig.unsqueeze(0)).detach().cpu().numpy()
    q_new = reloaded_trainer.agent.policy_net(z_new.unsqueeze(0)).detach().cpu().numpy()

    assert_true(np.allclose(q_orig, q_new, atol=1e-6), "Reloaded Q-values do not match original agent.")
    print(f"Reloaded Q-values match for task {sample_task.name}.")

    # 8. Post-train greedy evaluation
    print("\n" + "-" * 100)
    print("8. POST-TRAIN GREEDY EVALUATION")
    print("-" * 100)

    trainer.agent.policy_net.eval()
    trainer.agent.encoder.eval()
    trainer.agent.term_encoder.eval()

    greedy_rows = []
    for task in SMALL_TASKS:
        runtime = trainer.runtimes_by_task[task.task_id]
        current_terms = initial_terms_for_task(runtime, trainer.linear_additive)
        visited_state_keys = {runtime.action_manager.canonical_state_key(current_terms)}
        done = False
        H_tau = trainer.get_task_horizon(runtime)
        n_steps = 0

        while not done and n_steps < H_tau:
            action_index, _ = trainer.agent.select_action(
                current_terms=current_terms,
                action_manager=runtime.action_manager,
                runtime=runtime,
                visited_state_keys=visited_state_keys,
                epsilon=0.0,
                boltzmann=False,
                temperature=1.0,
            )
            current_terms, done = runtime.action_manager.apply_action(current_terms, action_index, visited_state_keys=visited_state_keys)
            visited_state_keys.add(runtime.action_manager.canonical_state_key(current_terms))
            n_steps += 1

        spec_key, reward = evaluate_final_terms(
            runtime=runtime,
            final_terms=current_terms,
            get_mnl_fn=trainer.get_mnl_fn,
            return_spec=True,
            suppress_exceptions=False,
            logger=LOGGER,
        )
        greedy_rows.append({
            "task_name": task.name,
            "specification": spec_key,
            "reward": reward,
            "n_steps": n_steps,
            "n_terms": len(current_terms),
        })
        print(f"[{task.name}] reward={reward:.4f} | steps={n_steps} | terms={len(current_terms)} | spec={spec_key}")

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df.to_csv(run_dir / "greedy_eval.csv", index=False)

    # 8b. Generate 10 Decisions proposals from the trained agent
    print("\n" + "-" * 100)
    print("8b. DECISIONS MODEL PROPOSALS")
    print("-" * 100)

    decisions_runtime = trainer.runtimes_by_task[DECISIONS_TASK.task_id]
    decisions_proposals_df, proposal_attempts = generate_decisions_proposals(
        trainer=trainer,
        runtime=decisions_runtime,
        n_proposals=N_DECISIONS_PROPOSALS,
        temperature=DECISIONS_PROPOSAL_TEMPERATURE,
        max_attempts=DECISIONS_PROPOSAL_MAX_ATTEMPTS,
    )

    assert_true(
        len(decisions_proposals_df) > 0,
        "The trained agent did not generate any Decisions proposal.",
    )

    decisions_proposals_df.to_csv(run_dir / "decisions_model_proposals.csv", index=False)
    print(
        decisions_proposals_df[
            [
                "proposal_id",
                "task_name",
                "specification",
                "reward",
                "n_steps",
                "n_terms",
                "forced_stop",
            ]
        ].to_string(index=False)
    )
    print(
        f"Generated {len(decisions_proposals_df)} unique Decisions proposals "
        f"in {proposal_attempts} attempts."
    )

    # 9. SQLite summaries
    print("\n" + "-" * 100)
    print("9. SQLITE DATABASE SUMMARY")
    print("-" * 100)

    sqlite_rows = []
    for task in SMALL_TASKS:
        sqlite_rows.append(inspect_rewards_sqlite(task))
    sqlite_df = pd.DataFrame(sqlite_rows)
    sqlite_df.to_csv(run_dir / "sqlite_summary.csv", index=False)
    print(sqlite_df.to_string(index=False))

    # 10. Save small summary
    print("\n" + "-" * 100)
    print("10. FINAL SUMMARY")
    print("-" * 100)

    summary = {
        "tasks": [t.name for t in SMALL_TASKS],
        "episode_count": int(trainer.episode_count),
        "learn_steps": int(trainer.learn_steps),
        "replay_size": int(len(trainer.replay_buffer)),
        "policy_changed": bool(policy_changed),
        "encoder_changed": bool(encoder_changed),
        "run_dir": str(run_dir),
        "latest_checkpoint": str(latest_ckpt),
        "elapsed_seconds": float(time.time() - started),
        "decisions_task": DECISIONS_TASK.name,
        "n_decisions_proposals": int(len(decisions_proposals_df)),
    }

    with open(run_dir / "small_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Decisions proposals to: {run_dir / 'decisions_model_proposals.csv'}")
    print(json.dumps(summary, indent=2))
    print("\nSMALL TRAIN PASSED.")


# =============================================================================
# Wrapper around get_mnl_outcomes with extra debugging
# =============================================================================
def trainer_get_mnl_outcomes_with_debug(*args, **kwargs):
    """
    Wrapper so small training prints richer debug information on failure.
    """
    try:
        return get_mnl_outcomes_debug_passthrough(*args, **kwargs)
    except Exception as exc:
        LOGGER.exception("get_mnl_outcomes failed during small test.")
        raise exc


def get_mnl_outcomes_debug_passthrough(*args, **kwargs):
    from env.environment import get_mnl_outcomes
    kwargs = dict(kwargs)
    kwargs.setdefault("debug_apollo", True)
    kwargs.setdefault("info", False)
    kwargs.setdefault("save", False)
    return get_mnl_outcomes(*args, **kwargs)


if __name__ == "__main__":
    main()