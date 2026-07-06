"""
env/apollo/estimator.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Apollo estimation backend.
"""

from __future__ import annotations
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Optional
import os
import subprocess
from .r_env import configure_r_environment
configure_r_environment()
import rpy2    
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, isinstalled
from .schema import ApolloSpecification
from mdp.task import Task



# ===================================
# Apollo loading
# ===================================

_APOLLO = None
_APOLLO_READY = False

def apollo_info() -> dict:
    return {
        "r_version": ro.r("R.version.string")[0],
        "r_home": ro.r('R.home("R")')[0],
        "libpaths": [str(x) for x in ro.r(".libPaths()")],
        "apollo_installed": isinstalled("apollo"),
        "apollo_loaded": _APOLLO is not None,
        "apollo_ready": _APOLLO_READY,
    }

def get_apollo():
    global _APOLLO
    if _APOLLO is None:
        _APOLLO = importr("apollo", suppress_messages=True)
    return _APOLLO

def ensure_apollo_ready() -> None:
    global _APOLLO_READY
    if _APOLLO_READY:
        return
    if not isinstalled("apollo"):
        raise RuntimeError("R package 'apollo' is not installed.")
    try:
        get_apollo()
        ro.r("suppressMessages(library(apollo))")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Apollo: {exc}") from exc
    _APOLLO_READY = True


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

def to_r_path(path_obj: Path) -> str:
    return str(path_obj.resolve()).replace("\\", "/")


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
# Summary extraction
# =============================================================================

def extract_model_summary(model, specification_key: str) -> pd.DataFrame:
    ro.globalenv["model"] = model
    ro.globalenv["specification_name"] = specification_key

    ro.r(
        """
        model_summary <- data.frame(
            specification = specification_name,
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
        """
    )

    summary_r = ro.globalenv["model_summary"]
    summary_dict = {}
    for col_name in summary_r.names:
        values = summary_r.rx2(col_name)
        if values is not None and len(values) > 0:
            summary_dict[col_name] = [values[0]]
        else:
            summary_dict[col_name] = [None]
    summary = pd.DataFrame(summary_dict)
    summary["skipped"] = 0
    return summary


# =============================================================================
# Main estimation runner
# =============================================================================

def run_apollo_estimation(
    task: Task,
    apollo_specification: ApolloSpecification,
    output_directory: Path,
    info: bool = False,
    save: bool = False,
    save_summary_file: bool = False,
    debug_apollo: bool = False,
    debug_path: Optional[Path] = None,
) -> pd.DataFrame:

    ensure_apollo_ready()
    validate_dataset(task.dataset_path, task)

    apollo = get_apollo()
    dataset_path = Path(task.dataset_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    if info:
        logger.info("Starting Apollo estimation: %s", apollo_specification.specification_key)

    try:
        # =====================================================
        # Apollo
        # =====================================================
        apollo.apollo_initialise()  

        # 1. Apollo control
        control =  {
            "modelName": apollo_specification.specification_key,
            "modelDescr": "MNL proposed by Delphos",
            "indivID": task.id_column,
            "outputDirectory": to_r_path(output_directory)
        }
        ro.globalenv["apollo_control"] = ro.ListVector(control)

        # 2. Apollo dataset
        ro.globalenv["database"] = ro.r(f'read.csv("{to_r_path(dataset_path)}", header=TRUE, sep=",")')

        # 3. Apollo parameters
        beta_values = list(apollo_specification.apollo_beta.values())
        beta_names = list(apollo_specification.apollo_beta.keys())
        apollo_beta = ro.FloatVector(beta_values)
        apollo_beta.names = ro.StrVector(beta_names)
        ro.globalenv["apollo_beta"] = apollo_beta
        ro.globalenv["apollo_fixed"] = ro.StrVector(apollo_specification.apollo_fixed)

        # 4. Apollo inputs
        apollo_inputs = apollo.apollo_validateInputs()

        # 5. Apollo probabilities
        apollo_probabilities = apollo_specification.probability_code
        ro.r(apollo_probabilities)
        
        # 6. Apollo estimation settings
        settings = ro.ListVector({"printLevel": 0, "writeIter": False, "silent": True})    
        ro.globalenv["estimate_settings"] = settings

        # Write debug files before running estimation to ensure they are saved on failure
        if debug_apollo and debug_path is not None:
            debug_path.mkdir(parents=True, exist_ok=True)
            write_debug_file(debug_path / "apollo_beta.txt", "\n".join(beta_names))
            write_debug_file(debug_path / "apollo_fixed.txt", "\n".join(apollo_specification.apollo_fixed))
            write_debug_file(debug_path / "apollo_probabilities.R", apollo_specification.probability_code)

        # 7. Model estimation
        model = apollo.apollo_estimate(
            apollo_beta,
            ro.globalenv["apollo_fixed"],
            ro.globalenv["apollo_probabilities"],
            apollo_inputs,
            settings)

    except Exception as exc:
        logger.exception("Estimation failed: %s", apollo_specification.specification_key)
        if debug_apollo and debug_path is not None:
            write_debug_file(debug_path / "error.txt", str(exc))
        raise

    if info:
        apollo.apollo_modelOutput(model)

    if save:
        apollo.apollo_saveOutput(model)

    summary = extract_model_summary(model, apollo_specification.specification_key,)

    if save_summary_file:
        summary_path = (output_directory / f"{apollo_specification.specification_key}_summary.csv")
        summary.to_csv(summary_path, index=False)
        if info:
            logger.info("Summary saved: %s", summary_path)
    return summary


def generate_apollo_r_script(
    task: Task,
    apollo_specification: ApolloSpecification,
    output_directory: Path,
    summary_file: Path,
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


    settings <- list(printLevel=0, writeIter=FALSE, silent=TRUE)

    model <- apollo_estimate(
        apollo_beta,
        apollo_fixed,
        apollo_probabilities,
        apollo_inputs,
        estimate_settings=settings
    )

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