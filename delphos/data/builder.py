from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from delphos.data.validation import validate_dataset_config
from delphos.grammar.task import Task


def create_dataset(
    folder: str | Path,
    *,
    name: str,
    csv_path: str | Path,
    choice_column: str,
    id_column: str = "id",
    panel: bool = False,
    alternatives: dict[str, dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    covariates: dict[str, dict[str, Any]] | None = None,
    ll_null: float | None = None,
    ll_linear: float | None = None,
    n_obs: int | None = None,
    dataset_id: int = 0,
) -> Task:
    """Create a Delphos dataset folder from a CSV and schema dictionaries."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)
    target_csv = folder / csv_path.name
    if csv_path.resolve() != target_csv.resolve():
        target_csv.write_bytes(csv_path.read_bytes())

    config = {
        "id": id_column,
        "choice": choice_column,
        "panel": bool(panel),
        "alternatives": alternatives,
        "attributes": attributes,
        "covariates": covariates or {},
        "path_choice_dataset": target_csv.name,
        "ll_null": ll_null,
        "ll_linear": ll_linear,
        "n_obs": n_obs,
        "df_name": name,
    }
    validate_dataset_config(config=config, csv_path=target_csv)
    yaml_path = folder / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return Task.from_yaml(id=dataset_id, yaml_path=yaml_path)
