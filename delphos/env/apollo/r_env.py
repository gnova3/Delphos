"""
env/apollo/r_environment.py

Utilities for discovering and configuring R before
loading rpy2.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def configure_r_environment() -> None:
    """
    Configure R_HOME and PATH before importing rpy2.

    Search order
    ------------
    1. Existing R_HOME
    2. `R RHOME`
    3. Common macOS Framework path
    4. Common Homebrew path

    Raises
    ------
    RuntimeError
        If R cannot be located.
    """
    r_home = os.environ.get("R_HOME")
    if r_home and Path(r_home).exists():
        return
    r_executable = shutil.which("R")
    if r_executable is not None:
        try:
            r_home = (subprocess.check_output([r_executable, "RHOME"],text=True,).strip())
            if Path(r_home).exists():
                os.environ["R_HOME"] = r_home
                r_bin = str(Path(r_executable).parent)
                if r_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = (f"{r_bin}:{os.environ.get('PATH', '')}")
                return
        except Exception:
            pass

    # macOS Framework R
    framework_r = Path("/Library/Frameworks/R.framework/Resources")
    if framework_r.exists():
        os.environ["R_HOME"] = str(framework_r)
        if "/usr/local/bin" not in os.environ.get("PATH", ""):
            os.environ["PATH"] = (
                "/usr/local/bin:"
                + os.environ.get("PATH", "")
            )
        return

    # macOS Apple Silicon
    homebrew_r = Path("/opt/homebrew/lib/R")
    if homebrew_r.exists():
        os.environ["R_HOME"] = str(homebrew_r)
        return
    raise RuntimeError(
        "Unable to locate an R installation. "
        "Please install R or define R_HOME."
    )