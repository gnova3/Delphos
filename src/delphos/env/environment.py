"""
env/environment.py

: author: Gabriel Nova
: date: Jun 2026
: version: 2.0.0
: purpose: Delphos estimation environment.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from delphos.grammar.task import Task
from delphos.env.result_cache import ResultCache
from delphos.env.apollo.schema import ApolloSpecification


# ==========================================================
# Evaluation
# ==========================================================
def evaluate_specification(
    task: Task,
    apollo_specification: ApolloSpecification,
    info: bool = False,
    save: bool = False,
    save_summary_file: bool = False,
    debug_apollo: bool = False,
    debug_path: Optional[Path] = None,
    raise_on_error: bool = False,
    max_free_parameters: Optional[int] = 50,
) -> pd.DataFrame:
    
    from delphos.env.apollo.estimator import run_apollo_estimation

    rewards_path = Path(task.rewards_path)    
    outputs_path = task.dataset_path.parent / "outputs"
    outputs_path.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(task.rewards_path)

    specification_key = apollo_specification.specification_key
    task_name = task.name

    # =====================================================
    # Write debug files early if debugging is enabled
    # =====================================================
    if debug_apollo and debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)
        beta_names = list(apollo_specification.apollo_beta.keys())
        (debug_path / "apollo_beta.txt").write_text("\n".join(beta_names), encoding="utf-8")
        (debug_path / "apollo_fixed.txt").write_text("\n".join(apollo_specification.apollo_fixed), encoding="utf-8")
        (debug_path / "apollo_probabilities.R").write_text(apollo_specification.probability_code, encoding="utf-8")

    # =====================================================
    # Cache lookup
    # =====================================================
    cached = cache.lookup(
        task_name=task_name, 
        specification=specification_key
    )
    if not cached.empty:
        is_success = int(cached.iloc[0]["successfulEstimation"]) == 1
        is_skipped = int(cached.iloc[0]["skipped"]) == 1
        # If it is a failed estimation and we are debugging, bypass cache to capture the actual error.txt
        if (is_success or is_skipped) or not debug_apollo:
            if info:
                print(f"Evaluated | {task_name} | {specification_key}")
            return cached

    # =====================================================
    # Skip large models
    # =====================================================
    n_free_parameters = (len(apollo_specification.apollo_beta)
                        - len(apollo_specification.apollo_fixed))

    if (max_free_parameters is not None and n_free_parameters > max_free_parameters):
        outcome = cache.skipped(
            task_name=task_name,
            specification=specification_key,
            n_free_parameters=n_free_parameters,
        )
        cache.upsert(outcome)
        if info:
            print(f"Skipped | {task_name} | {specification_key}")            
        return outcome

    # =====================================================
    # Apollo estimation
    # =====================================================

    try:
        outcome = run_apollo_estimation(
            task=task,
            apollo_specification=apollo_specification,
            output_directory=outputs_path,
            info=info,
            save=save,
            save_summary_file=save_summary_file,
            debug_apollo=debug_apollo,
            debug_path=debug_path,
        )
    except Exception as e:
        print(f"EXCEPTION CAUGHT IN evaluate_specification: {e}")
        import traceback
        traceback.print_exc()
        outcome = cache.failed(task_name=task_name, specification=specification_key)
        cache.upsert(outcome)
        if raise_on_error:
            raise
        return outcome

    # =====================================================
    # Cache result
    # =====================================================

    outcome["task_name"] = task_name
    outcome["specification"] = specification_key
    cache.upsert(outcome)
    return outcome

