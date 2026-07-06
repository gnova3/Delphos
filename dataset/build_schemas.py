import pandas as pd
import yaml
from pathlib import Path
import shutil

def load_dataset_registry(datasets: dict, base_path: Path) -> dict:
    registry = {}

    for dataset_id, dataset_name in datasets.items():
        processed_path = base_path / dataset_name / "processed"
        metadata_path = processed_path / "dataset.yaml"

        if not metadata_path.exists():
            print(f"dataset {dataset_id} ({dataset_name}) -> Missing dataset.yaml, skipping")
            continue

        # Look for the data CSV
        csv_files = list(processed_path.glob("*.csv"))
        valid_files = [f for f in csv_files if "dictionary" not in f.name.lower()]
        
        # Prefer non-formatted files for metadata extraction if possible
        non_formatted = [f for f in valid_files if not f.name.endswith("_formatted.csv")]
        target_csv = non_formatted[0] if non_formatted else (valid_files[0] if valid_files else None)

        if not target_csv:
            print(f"dataset {dataset_id} ({dataset_name}) -> No CSV found, skipping")
            continue

        df = pd.read_csv(target_csv)

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        registry[dataset_id] = {"dataset_name": dataset_name, "metadata": metadata, "dataframe": df}
    return registry

def build_schema(meta: dict, dataset_id: int, output_path: Path) -> dict:

    schema = {
        "id": meta["data_structure"]["id"],
        "choice": meta["data_structure"]["choice"],
        "panel": meta["data_structure"]["panel"],
        "alternatives": {a["name"]: {"id": a["id"], "availability": a.get("availability")} for a in meta["alternatives"]},
        "attributes": meta.get("attributes", {}),
        "covariates": meta.get("covariates", {})
    }

    out_dir = output_path / f"dataset_{dataset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "dataset.yaml"

    if out_file.exists():
        print(
            f"dataset {dataset_id} -> {out_file} already exists, "
            "skipping to preserve manual modifications."
        )
        return None
    with open(out_file, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    print(f"dataset {dataset_id} -> Created schema at {out_file}")
    return schema

def build_schemas(dataset_registry: dict, output_path: Path):
    for dataset_id, entry in dataset_registry.items():
        build_schema(entry["metadata"], dataset_id, output_path)

def copy_datasets_to_delphos(datasets: dict, base_path: Path, output_path: Path,):

    for dataset_id, dataset_name in datasets.items():
        source_dir = base_path / dataset_name / "processed"
        target_dir = output_path / f"dataset_{dataset_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_files = [f for f in source_dir.glob("*.csv") if "dictionary" not in f.name.lower()]
        
        if not csv_files:
            print(f"dataset {dataset_id} -> No CSV found")
            continue

        if len(csv_files) > 1:
            print(f"dataset {dataset_id} -> Multiple CSVs found, using {csv_files[0].name}")

        source_csv = csv_files[0]
        target_csv = target_dir / source_csv.name

        shutil.copy2(source_csv, target_csv)

        print(f"dataset {dataset_id} -> Copied {source_csv.name}")