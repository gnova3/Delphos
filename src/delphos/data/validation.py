from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def validate_dataset_config(config: dict[str, Any], csv_path: str | Path) -> None:
    """Validate that a Delphos dataset schema matches its CSV columns."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Choice dataset not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Choice dataset is empty: {csv_path}") from exc

    available_columns = set(header)
    required_columns = {str(config["id"]), str(config["choice"])}

    for alternative in config.get("alternatives", {}).values():
        availability = alternative.get("avail", alternative.get("availability"))
        if availability is not None:
            required_columns.add(str(availability))

    for attribute in config.get("attributes", {}).values():
        mapping = attribute.get("mapping", {})
        required_columns.update(str(column) for column in mapping.values())

    for covariate in config.get("covariates", {}).values():
        source = covariate.get("source")
        if source is not None:
            required_columns.add(str(source))

    missing = sorted(required_columns - available_columns)
    if missing:
        raise ValueError(
            "Choice dataset is missing required columns: "
            + ", ".join(missing)
        )

    _validate_ids(config)


def _validate_ids(config: dict[str, Any]) -> None:
    attribute_ids = [int(spec["id"]) for spec in config.get("attributes", {}).values()]
    covariate_ids = [
        int(spec["id"])
        for spec in config.get("covariates", {}).values()
        if spec.get("id") is not None
    ]
    alternative_ids = [int(spec["id"]) for spec in config.get("alternatives", {}).values()]

    _ensure_unique(attribute_ids, "attribute")
    _ensure_unique(covariate_ids, "covariate")
    _ensure_unique(alternative_ids, "alternative")


def _ensure_unique(values: list[int], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} ids detected.")
