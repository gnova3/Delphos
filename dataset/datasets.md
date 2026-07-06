# Managing the `transport-choice-datasets` Submodule

This guide explains how to integrate and manage the `transport-choice-datasets` repository as a submodule within the `Delphos-core` repository.

## 1. Adding the Submodule

To bring the [transport-choice-datasets](https://github.com/TUD-CityAI-Lab/transport-choice-datasets) repository into `Delphos-core` as a submodule, run the following command from the root of the `Delphos-core` repository.

```bash
git submodule add https://github.com/TUD-CityAI-Lab/transport-choice-datasets.git
```

This command will:

1. Clone the repository into the specified directory (`./transport-choice-datasets`).
2. Add a `.gitmodules` file (or update it if it already exists) to track the submodule.
3. Stage both the `.gitmodules` file and the submodule directory for your next commit.

**Remember to commit these changes:**

```bash
git commit -m "Add transport-choice-datasets as a submodule"
```

## 2. Cloning the Repository with the Submodule

When other developers clone the `Delphos-core` repository, the submodule directory will be empty by default.

**To clone the repository and initialize the submodule at the same time:**

```bash
git clone --recurse-submodules https://github.com/gnova3/Delphos-core.git
```

**If the repository was already cloned without the submodule:**
You can initialize and fetch the submodule data by running:

```bash
git submodule update --init --recursive
```

## 3. Updating the Submodule

If the upstream `transport-choice-datasets` repository is updated and you want to pull those latest changes into your submodule, use:

```bash
git submodule update --remote transport-choice-datasets
```

After updating, the submodule will point to a new commit. You must commit this pointer change in your main `Delphos-core` repository:

```bash
git add transport-choice-datasets
git commit -m "Update transport-choice-datasets submodule"
```

## 4. Dataset Preparation Workflow (Building Schemas)

When new datasets are pulled from the submodule, they must be formatted into Delphos nomenclature. This process is managed via the `datasets.ipynb` notebook and relies on functions defined in `build_schemas.py`.

The general workflow follows these four steps:

### Step 1: Load Dataset Registry

The process starts by mapping raw dataset names to integer IDs using the `DATASETS` dictionary. Running `load_dataset_registry()` scans the submodule directories, locates the unformatted `.csv` data, and loads the upstream `dataset.yaml` metadata.

### Step 2: Build Initial Schemas

Calling `build_schemas()` automatically generates new `dataset_X/dataset.yaml` files in the Delphos `dataset/` directory. These files contain the structural skeleton for the dataset, capturing its attributes, covariates, and alternatives based on the upstream metadata. To prevent accidental data loss, this script skips any `dataset.yaml` files that already exist.

### Step 3: Manually Assign Attribute IDs

Once the initial YAML schemas are generated, you must open each newly created `dataset_X/dataset.yaml` file and manually assign integer IDs to the attributes. For example, if the schema lists `time:` and `cost:`, you must update them to include `id: 1` and `id: 2`, respectively.

### Step 4: Reformat the Dataset

Finally, run `reformat_datasets_for_delphos()`. This function reads your modified `dataset.yaml` schema and translates the original source dataset into Delphos-compatible nomenclature. It specifically standardizes columns to format:

- Attributes as `x_{alternative_id}_{attribute_id}`
- Availabilities as `av_{alternative_id}`
- Decision target as `choice`
- Agent IDs as `id`

The reformatted dataset is then saved as a new `{dataset_name}_formatted.csv` file within the local `dataset_X` folder, ready for training!
