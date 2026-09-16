"""Final-user Delphos package."""

from delphos.api import (
    DelphosModel,
    configure_modelling_space,
    create_dataset,
    list_datasets,
    load_agent,
    load_dataset,
    propose_models,
    set_covariate_levels,
    validate_dataset_config,
)

__all__ = [
    "DelphosModel",
    "configure_modelling_space",
    "create_dataset",
    "list_datasets",
    "load_agent",
    "load_dataset",
    "propose_models",
    "set_covariate_levels",
    "validate_dataset_config",
]

import subprocess
import warnings
import threading

def _check_r_environment():
    try:
        r_code = """
        r_version <- R.version.string
        if (requireNamespace('apollo', quietly = TRUE)) {
            apollo_version <- as.character(packageVersion('apollo'))
            cat(sprintf('Delphos Backend: %s | Apollo v%s', r_version, apollo_version))
        } else {
            quit(status=1)
        }
        """
        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            warnings.warn(
                "R is installed, but the 'apollo' package is missing. "
                "Delphos can still propose models, but estimation will fail. "
                "Please run `install.packages('apollo')` in R.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # Print the successful version string
            print(result.stdout.strip())
            
    except FileNotFoundError:
        warnings.warn(
            "Rscript was not found on your system PATH. "
            "Delphos can still propose models, but estimation will fail.",
            UserWarning,
            stacklevel=2,
        )

# Run the check synchronously so the print output shows up properly in Jupyter Notebooks
_check_r_environment()
