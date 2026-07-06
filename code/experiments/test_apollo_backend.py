from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.task_registry import TASKS
from mdp.task import Task
from mdp.state_builder import TaskSpecification, StateLayoutBuilder
from mdp.state_manager import StateManager, SpecificationTerm
from env.environment import get_mnl_outcomes
from configs.task_configuration import GLOBAL_ATTRIBUTE_IDS, GLOBAL_COVARIATE_IDS

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_state_manager(task: Task) -> StateManager:
    builder = StateLayoutBuilder(
        dataset_schema=task.dataset_schema,
        global_attribute_ids=GLOBAL_ATTRIBUTE_IDS,
        global_covariate_ids=GLOBAL_COVARIATE_IDS,
        device="cpu",
    )

    task_spec = TaskSpecification(
        attribute_names=task.attribute_names,
        covariate_names=task.covariate_names,
        covariate_levels=task.covariate_levels,
        transformation_names=tuple(task.dataset_schema.transformation_names)
        if task.dataset_schema.has_transformations else None,
        taste_names=tuple(task.dataset_schema.taste_names)
        if task.dataset_schema.has_tastes else None,
    )

    layout = builder.build(task_spec)

    return StateManager(
        dataset_schema=task.dataset_schema,
        layout=layout,
        device="cpu",
    )

def make_backend_key(task: Task, terms: list[SpecificationTerm]) -> tuple[str, list[int], list[int], list[int], StateManager]:
    state_manager = build_state_manager(task)
    state = state_manager.encode_terms(terms)
    backend = state_manager.to_backend_specification(state)
    return (
        backend.key,
        backend.transformation_ids,
        backend.taste_ids,
        backend.covariate_ids,
        state_manager,
    )


def candidate_specs_for_task(task: Task) -> list[list[SpecificationTerm]]:
    attr_ids = list(task.attribute_ids)
    cov_ids = list(task.covariate_ids)

    has_cov = len(cov_ids) > 0
    c1 = cov_ids[0] if has_cov else 0
    c2 = cov_ids[1] if len(cov_ids) > 1 else c1 if has_cov else 0

    out: list[list[SpecificationTerm]] = []

    # 1. ASC only
    out.append([
        SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0)
    ])

    # 2. Linear generic on first real attribute
    if len(attr_ids) > 1:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=1, taste_id=1, covariate_id=0),
        ])

    # 3. Linear specific on first two real attributes
    if len(attr_ids) > 2:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=1, taste_id=2, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[2], transformation_id=1, taste_id=2, covariate_id=0),
        ])

    # 4. Log generic with covariate
    if len(attr_ids) > 1:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=2, taste_id=1, covariate_id=c1),
        ])

    # 5. Mixed specification
    if len(attr_ids) > 3:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[2], transformation_id=2, taste_id=2, covariate_id=c1),
            SpecificationTerm(attribute_id=attr_ids[3], transformation_id=1, taste_id=1, covariate_id=c2 if has_cov else 0),
        ])

    # 6. Box-Cox generic
    if len(attr_ids) > 1:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=3, taste_id=1, covariate_id=0),
        ])

    # 7. Maximum Complexity (All attributes Box-Cox, specific, with rotating covariates)
    if len(attr_ids) > 1:
        max_complex = [
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0)
        ]
        for i, attr_id in enumerate(attr_ids[1:]):
            cov_to_use = cov_ids[i % len(cov_ids)] if has_cov else 0
            max_complex.append(
                SpecificationTerm(attribute_id=attr_id, transformation_id=3, taste_id=2, covariate_id=cov_to_use)
            )
        out.append(max_complex)

    # 8. No ASC, only linear generic (tests if model runs without base constants)
    if len(attr_ids) > 1:
        out.append([
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=1, taste_id=1, covariate_id=0)
        ])

    # 9. Invalid/Out-of-bounds (Deliberately bad specification to test graceful failure)
    # We use a transformation ID of 999 which should be caught before R execution.
    if len(attr_ids) > 1:
        out.append([
            SpecificationTerm(attribute_id=1, transformation_id=1, taste_id=1, covariate_id=0),
            SpecificationTerm(attribute_id=attr_ids[1], transformation_id=999, taste_id=1, covariate_id=0),
        ])

    return out


def run_one(task: Task, terms: list[SpecificationTerm], debug_apollo: bool = False) -> dict:
    key, transform_ids, taste_ids, covariate_ids, state_manager = make_backend_key(task, terms)

    result = get_mnl_outcomes(
        apollo_str=key,
        transform_0=transform_ids,
        taste_0=taste_ids,
        cov_0=covariate_ids,
        state_manager=state_manager,
        path_rewards=task.path_rewards,
        path_choice_dataset=task.path_choice_dataset,
        info=True,
        save=False,
        debug_apollo=debug_apollo,
        raise_on_error=False,
    )

    row = result.tail(1).iloc[0].to_dict()
    row["task_name"] = task.name
    row["specification"] = key
    return row


def print_summary(rows: list[dict], save_path: Path | None = None) -> None:
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("Apollo backend smoke test summary")
    lines.append("=" * 100)

    for row in rows:
        ok = bool(row.get("successfulEstimation", False))
        llout = row.get("LLout", None)
        lines.append(
            f"{row['task_name']:<24} | "
            f"success={str(ok):<5} | "
            f"LLout={str(llout):<20} | "
            f"spec={row['specification']}"
        )

    lines.append("-" * 100)
    n_success = sum(bool(row.get("successfulEstimation", False)) for row in rows)
    lines.append(f"Successful estimations: {n_success}/{len(rows)}")

    summary_text = "\n".join(lines)
    print(summary_text)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(summary_text, encoding="utf-8")
        print(f"\nSummary saved to: {save_path}")


def main() -> None:
    run_name = f"iteration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repo_root = PROJECT_ROOT.parent
    save_dir = repo_root / "experiments" / "apollo_test" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Capture all print output to a log file
    log_file_path = save_dir / "test_log.txt"
    f = open(log_file_path, "w", encoding="utf-8")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for file in self.files:
                file.write(obj)
                file.flush()
        def flush(self):
            for file in self.files:
                file.flush()
                
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    try:
        rows: list[dict] = []

        for task in TASKS:
            print("\n" + "=" * 100)
            print(f"Testing task: {task.name}")
            print("=" * 100)

            specs = candidate_specs_for_task(task)

            for i, terms in enumerate(specs, start=1):
                print(f"\n--- Candidate {i}/{len(specs)} for {task.name} ---")
                
                # Identify if this is the deliberately bad spec
                is_deliberately_bad = any(t.transformation_id == 999 for t in terms)
                
                try:
                    row = run_one(task, terms, debug_apollo=True)
                    if is_deliberately_bad:
                        row["successfulEstimation"] = False
                        row["specification"] += " (WARNING: Expected failure but succeeded)"
                except Exception as exc:
                    row = {
                        "task_name": task.name,
                        "successfulEstimation": False,
                        "LLout": None,
                        "specification": f"EXCEPTION (Expected) {exc}" if is_deliberately_bad else f"EXCEPTION: {exc}",
                    }
                rows.append(row)

        print_summary(rows, save_path=save_dir / "test_summary.txt")
        print(f"\nFull console output saved to: {log_file_path}")
    finally:
        sys.stdout = original_stdout
        f.close()


if __name__ == "__main__":
    main()