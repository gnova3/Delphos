import pandas as pd
import yaml
from pathlib import Path

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
    # This matches the logic from the notebook
    schema = {
        "id": meta["data_structure"]["id"],
        "choice": meta["data_structure"]["choice"],
        "panel": meta["data_structure"]["panel"],
        
        "alternatives": {
            a["name"]: {"id": a["id"], "avail": a.get("availability")} 
            for a in meta["alternatives"]
        },

        "attributes": {
            attr_name: {"id": None,  "mapping": attr_info["by_alternative"]} 
            for attr_name, attr_info in meta.get("attributes", {}).items()
        },

        "covariates": {
            cov_name: {"id": None, "source": cov_info.get("source", cov_name), "type": cov_info.get("type"), "levels": cov_info.get("levels")}
            for cov_name, cov_info in meta.get("covariates", {}).items()
        }
    }    
    
    out_dir = output_path / f"dataset_{dataset_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset.yaml"
    
    # CRITICAL: Skip if the file already exists to avoid overwriting manual changes
    if out_file.exists():
        print(f"dataset {dataset_id} -> {out_file} already exists, skipping to preserve manual modifications.")
        return None
    
    with open(out_file, "w") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    
    print(f"dataset {dataset_id} -> Created schema at {out_file}")
    return schema

def build_schemas(dataset_registry: dict, output_path: Path):
    for dataset_id, entry in dataset_registry.items():
        build_schema(entry["metadata"], dataset_id, output_path)

def reformat_datasets_for_delphos(datasets: dict, base_path: Path, output_path: Path):
    """
    Reformats original datasets into Delphos nomenclature:
    - x_{alternative_id}_{attribute_id}
    - av_{alternative_id}
    - choice
    - id
    """
    for dataset_id, dataset_name in datasets.items():
        # 1. Load the schema (dataset.yaml)
        schema_path = output_path / f"dataset_{dataset_id}" / "dataset.yaml"
        if not schema_path.exists():
            print(f"dataset {dataset_id} -> Missing dataset.yaml in {schema_path}, skipping")
            continue
            
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
            
        # 2. Find and load the original CSV
        processed_path = base_path / dataset_name / "processed"
        csv_files = list(processed_path.glob("*.csv"))
        
        # Filter out dictionary files and previously formatted files
        valid_files = [f for f in csv_files if "dictionary" not in f.name.lower()]
        
        # Prefer non-formatted files if available
        non_formatted = [f for f in valid_files if not f.name.endswith("_formatted.csv")]
        
        if non_formatted:
            target_csv = non_formatted[0]
        elif valid_files:
            target_csv = valid_files[0]
        else:
            print(f"dataset {dataset_id} -> No valid CSV found in {processed_path} (filtered dictionaries and formatted files), skipping")
            continue
        
        print(f"dataset {dataset_id} -> Processing {target_csv.name}")
        df = pd.read_csv(target_csv)
        
        # 3. Create the mapping
        new_data = {}
        
        # ID and Choice
        id_col = schema.get("id")
        choice_col = schema.get("choice")
        
        if id_col in df.columns:
            new_data["id"] = df[id_col]
        else:
            print(f"  Warning: ID column '{id_col}' not found")
            
        if choice_col in df.columns:
            new_data["choice"] = df[choice_col]
        else:
            print(f"  Warning: Choice column '{choice_col}' not found")
            
        # Alternatives and Availability
        alts = schema.get("alternatives", {})
        for alt_name, alt_info in alts.items():
            alt_id = alt_info.get("id")
            avail_col = alt_info.get("avail")
            if avail_col:
                found_col = None
                for col in df.columns:
                    if col.lower() == avail_col.lower() or col.lower() == avail_col.lower().replace("_", ""):
                        found_col = col
                        break
                
                if found_col:
                    new_data[f"av_{alt_id}"] = df[found_col]
                else:
                    print(f"  Warning: Availability column '{avail_col}' not found for alternative '{alt_name}'")
        
        # Attributes mapping: x_{alt_id}_{attr_id}
        attrs = schema.get("attributes", {})
        for i, (attr_name, attr_info) in enumerate(attrs.items()):
            # If attr_id is missing/null, assign a sequential one (starting at 1)
            attr_id = attr_info.get("id")
            if attr_id is None:
                attr_id = i + 1
            
            mapping = attr_info.get("mapping", {})
            for alt_name, original_col in mapping.items():
                if alt_name not in alts:
                    continue
                alt_id = alts[alt_name].get("id")
                
                if original_col in df.columns:
                    new_data[f"x_{alt_id}_{attr_id}"] = df[original_col]
                else:
                    # Fill with 0 for missing alternatives/attributes
                    new_data[f"x_{alt_id}_{attr_id}"] = 0
        
        # Covariates
        covs = schema.get("covariates", {})
        for cov_name, cov_info in covs.items():
            source_col = cov_info.get("source", cov_name)
            if source_col in df.columns:
                new_data[cov_name] = df[source_col]
        
        # 4. Save the formatted CSV
        if not new_data:
            print(f"dataset {dataset_id} -> No columns mapped, skipping")
            continue

        formatted_df = pd.DataFrame(new_data)
        out_file = output_path / f"dataset_{dataset_id}" / f"{dataset_name}_formatted.csv"
        formatted_df.to_csv(out_file, index=False)
        print(f"dataset {dataset_id} -> Saved formatted CSV: {out_file.name}")
