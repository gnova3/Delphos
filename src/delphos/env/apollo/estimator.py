"""
env/apollo/estimator.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Apollo estimation backend (Rscript execution).
"""

from __future__ import annotations
import pandas as pd
import logging
import sys
import shutil
from pathlib import Path
from typing import Optional
import subprocess

from .schema import ApolloSpecification
from delphos.grammar.task import Task


# =============================================================================
# Logging
# =============================================================================

def _bootstrap_delphos_logger() -> logging.Logger:
    logger_root = logging.getLogger("Delphos")
    if getattr(logger_root, "_configured_console", False):
        return logger_root
    logger_root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt)
    logger_root.addHandler(handler)
    logger_root.propagate = False
    logger_root._configured_console = True
    return logger_root


_bootstrap_delphos_logger()
logger = logging.getLogger("Delphos.apollo")
logger.setLevel(logging.INFO)


# =============================================================================
# Utilities
# =============================================================================

def write_debug_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_dataset(dataset_path: str | Path, task: Task) -> None:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Choice dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, received: {dataset_path}")

    required = {"id", task.choice_column}
    for alt in task.alternatives:
        if alt.availability is not None:
            required.add(alt.availability)
    header = pd.read_csv(dataset_path, nrows=0)
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")


# =============================================================================
# Main estimation runner
# =============================================================================

def run_apollo_estimation(
    task: Task,
    apollo_specification: ApolloSpecification,
    output_directory: Path,
    info: bool = True,
    save: bool = True,
    save_summary_file: bool = True,
    debug_apollo: bool = False,
    debug_path: Optional[Path] = None,
) -> pd.DataFrame:

    if not shutil.which("Rscript"):
        raise RuntimeError("Rscript executable not found in PATH. Please install R to run Apollo estimation.")

    validate_dataset(task.dataset_path, task)
    output_directory.mkdir(parents=True, exist_ok=True)

    spec_key = apollo_specification.specification_key
    if info:
        logger.info("Starting Apollo estimation: %s", spec_key)

    summary_path = output_directory / f"{spec_key}_summary.csv"
    r_script_path = output_directory / f"{spec_key}_estimation.R"

    r_code = generate_apollo_r_script(
        task=task,
        apollo_specification=apollo_specification,
        output_directory=output_directory,
        summary_file=summary_path,
        save=save,
        info=info,
    )

    if debug_apollo and debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)
        beta_names = list(apollo_specification.apollo_beta.keys())
        write_debug_file(debug_path / "apollo_beta.txt", "\n".join(beta_names))
        write_debug_file(debug_path / "apollo_fixed.txt", "\n".join(apollo_specification.apollo_fixed))
        write_debug_file(debug_path / "apollo_probabilities.R", apollo_specification.probability_code)
        write_debug_file(debug_path / "estimation.R", r_code)

    r_script_path.write_text(r_code, encoding="utf-8")

    try:
        result = subprocess.run(
            ["Rscript", str(r_script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        if info:
            print(result.stdout)
    except subprocess.CalledProcessError as exc:
        logger.exception("Estimation failed: %s", spec_key)
        if debug_apollo and debug_path is not None:
            write_debug_file(debug_path / "error.txt", exc.stderr or exc.stdout)
            write_debug_file(debug_path / "stdout.txt", exc.stdout)
        raise RuntimeError(f"Rscript failed during estimation:\n{exc.stderr}") from exc


    if not summary_path.exists():
        raise RuntimeError(f"Estimation completed but summary file {summary_path} was not created.")

    summary = pd.read_csv(summary_path)
    summary["skipped"] = 0

    if info:
        logger.info("Summary saved: %s", summary_path)
    


    return summary


def generate_apollo_r_script(
    task: Task,
    apollo_specification: ApolloSpecification,
    output_directory: Path,
    summary_file: Path,
    save: bool = True,
    info: bool = True,
) -> str:

    dataset_path = str(Path(task.dataset_path).resolve()).replace("\\", "/")
    output_directory = str(Path(output_directory).resolve()).replace("\\", "/")
    summary_file = str(Path(summary_file).resolve()).replace("\\", "/")

    beta_lines = []
    for name, value in apollo_specification.apollo_beta.items():
        beta_lines.append(f'"{name}"={value}')

    beta_code = ",\n    ".join(beta_lines)

    fixed_code = ", ".join(
        [f'"{x}"' for x in apollo_specification.apollo_fixed]
    )
    
    save_output_code = "apollo_saveOutput(model)" if save else ""

    script = f'''
    suppressMessages(library(apollo))

    apollo_initialise()

    apollo_control <- list(
        modelName="{apollo_specification.specification_key}",
        modelDescr="MNL proposed by Delphos",
        indivID="{task.id_column}",
        outputDirectory="{output_directory}"
    )

    database_1 <- read.csv("{dataset_path}", header=TRUE, sep=",")
    set.seed(123)
    individuals <- unique(database_1$id)
    n_individuals <- length(individuals)
    train_individuals <- sample(individuals, size=0.8*n_individuals)
    test_individuals <- setdiff(individuals,train_individuals)
    database_1$test <- ifelse(database_1$id %in% test_individuals, 1, 0)
    in_sample <- subset(database_1,test==0)
    database <- in_sample

    apollo_beta <- c({beta_code})
    apollo_fixed <- c({fixed_code})

    apollo_inputs <- apollo_validateInputs()

    {apollo_specification.probability_code}


    settings <- list(printLevel=ifelse({str(info).upper()}, 3, 0), writeIter=FALSE, silent=ifelse({str(info).upper()}, FALSE, TRUE))

    model <- apollo_estimate(
        apollo_beta,
        apollo_fixed,
        apollo_probabilities,
        apollo_inputs,
        estimate_settings=settings
    )
    
    {save_output_code}

    summary_df <- data.frame(
        specification="{apollo_specification.specification_key}",
        numParams = model$numParams,
        numResids = model$numResids,
        maximum = model$maximum,
        vcHessianConditionNumber = model$vcHessianConditionNumber,
        successfulEstimation = model$successfulEstimation,
        LL0 = model$LL0,
        LLC = model$LLC,
        LLout = model$LLout,
        rho2_0 = model$rho2_0,
        adjRho2_0 = model$adjRho2_0,
        rho2_C = model$rho2_C,
        adjRho2_C = model$adjRho2_C,
        AIC = model$AIC,
        BIC = model$BIC,
        eigValue = model$eigValue[1],
        timeTaken = model$timeTaken,
        nFreeParams = model$nFreeParams
    )

    write.csv(summary_df, "{summary_file}", row.names=FALSE)
    '''
    return script
