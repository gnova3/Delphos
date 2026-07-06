from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIRS = [
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_1_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_2_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_3_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_4_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_5_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_6_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_7_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_8_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/dqn_single_agent_task_9_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/episode_full_agent_task_1_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_10_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_1_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_2_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_3_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_4_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_5_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/full_agent_task_6_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_1_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_2_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_3_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_4_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_5_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_6_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_7_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/checkpoints/single_agent_task_8_seed_123"),
    Path("/Users/gnova/Developer/Delphos-core/experiments/small_full_pipeline/iteration_20260526_220728"),
]


def compute_spec_diversity(specs: list[str], top_k: int = 5) -> dict:
    specs = [str(s) for s in specs if pd.notna(s)]
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
    n_total = len(specs)
    n_unique = len(counter)
    probs = np.array([count / n_total for count in counter.values()], dtype=float)

    return {
        "n_unique": int(n_unique),
        "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
        "top_k_specs": counter.most_common(top_k),
        "repeated_ratio": float((n_total - n_unique) / n_total),
        "novelty_ratio": float(n_unique / n_total),
        "n_total": int(n_total),
    }


def load_existing_diversity(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def spec_column(episode_df: pd.DataFrame) -> str:
    for column in ("specification_key", "spec_key"):
        if column in episode_df.columns:
            return column
    raise ValueError("No specification key column found.")


def group_key(group: pd.DataFrame, existing: dict) -> str:
    task_id = str(group["task_id"].iloc[0]) if "task_id" in group.columns else None
    task_name = str(group["task_name"].iloc[0]) if "task_name" in group.columns else None

    for candidate in (task_name, task_id):
        if candidate is not None and candidate in existing:
            return candidate

    if task_id is not None:
        return task_id
    if task_name is not None:
        return task_name
    return "all"


def task_groups(episode_df: pd.DataFrame):
    if "task_id" in episode_df.columns:
        return episode_df.groupby("task_id", sort=True)
    if "task_name" in episode_df.columns:
        return episode_df.groupby("task_name", sort=True)
    return [("all", episode_df)]


def regenerate_run(run_dir: Path) -> dict:
    episode_path = run_dir / "per_episode_diagnostics.csv"
    diversity_path = run_dir / "diversity_summary.json"

    episode_df = pd.read_csv(episode_path)
    existing = load_existing_diversity(diversity_path)
    spec_col = spec_column(episode_df)

    summary = {}
    for _, group in task_groups(episode_df):
        key = group_key(group, existing)
        spec_keys = group[spec_col].astype(str).tolist()
        spec_diversity = compute_spec_diversity(spec_keys)
        spec_diversity.pop("top_k_specs", None)

        old_task_summary = existing.get(key, {})
        old_spec_div = old_task_summary.get("spec_diversity", {})
        if "avg_pairwise_hamming" in old_spec_div:
            spec_diversity["avg_pairwise_hamming"] = old_spec_div["avg_pairwise_hamming"]

        summary[key] = {
            "spec_diversity": spec_diversity,
            "embedding_stats": old_task_summary.get("embedding_stats", {}),
        }

    diversity_path.write_text(
        json.dumps(clean_for_json(summary), indent=2),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "tasks": len(summary),
        "output": str(diversity_path),
    }


def main() -> None:
    results = []
    for run_dir in RUN_DIRS:
        results.append(regenerate_run(run_dir))

    for result in results:
        print(
            f"Regenerated {result['tasks']:>2} task group(s): "
            f"{result['output']}"
        )


if __name__ == "__main__":
    main()
