from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd


def get_delphos_logger(name: str = "Delphos") -> logging.Logger:
    """Get or create a configured Delphos logger.

    Args:
        name (str, optional): The name of the logger. Defaults to "Delphos".

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured_console", False):
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False
    logger._configured_console = True
    return logger


def create_run_directory(root: str = "experiments/run", prefix: str = "iteration") -> str:
    """Create a unique run directory.

    Example: experiments/run/iteration_20260413_101530

    Args:
        root (str, optional): The root directory. Defaults to "experiments/run".
        prefix (str, optional): The prefix for the directory. Defaults to "iteration".

    Returns:
        str: The path to the created unique directory.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root_path / f"{prefix}_{timestamp}"

    counter = 1
    while candidate.exists():
        candidate = root_path / f"{prefix}_{timestamp}_{counter}"
        counter += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return str(candidate)


def attach_file_logger(
    logger: logging.Logger,
    log_dir: str,
    filename: str = "agent_log.txt",
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
) -> None:
    """Attach a rotating file handler to the specified logger.

    Args:
        logger (logging.Logger): The logger instance.
        log_dir (str): The directory to save the log file.
        filename (str, optional): The name of the log file. Defaults to "agent_log.txt".
        max_bytes (int, optional): Maximum bytes per file before rotating. Defaults to 2_000_000.
        backup_count (int, optional): Number of backup files to keep. Defaults to 3.
    """
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = os.path.join(log_dir, filename)

    # Remove previous rotating file handlers pointing to the same file
    kept_handlers = []
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == os.path.abspath(path):
                    handler.close()
                    continue
            except Exception:
                pass
        kept_handlers.append(handler)

    logger.handlers = kept_handlers

    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary as a JSON file.

    Args:
        data (dict): The data dictionary to save.
        path (str | Path): The destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def flush_records(
    subfolder: Optional[str],
    training_log: list[dict],
    buffer_log: list[dict],
    enable_parquet_logs: bool = False,
    enable_csv_fallback: bool = True,
    parquet_compression: str = "snappy",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Flush the accumulated training and buffer logs to disk.

    Supports saving as Parquet (preferred) or CSV format.

    Args:
        subfolder (Optional[str]): The output directory.
        training_log (list[dict]): The accumulated training log records.
        buffer_log (list[dict]): The accumulated buffer log records.
        enable_parquet_logs (bool, optional): Whether to attempt saving to Parquet. Defaults to False.
        enable_csv_fallback (bool, optional): Whether to save to CSV as fallback. Defaults to True.
        parquet_compression (str, optional): Compression format for Parquet. Defaults to "snappy".
        logger (Optional[logging.Logger], optional): Logger instance for warnings. Defaults to None.
    """
    if not subfolder:
        return

    subfolder_path = Path(subfolder)
    subfolder_path.mkdir(parents=True, exist_ok=True)

    try:
        if enable_parquet_logs:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                if training_log:
                    pq.write_table(
                        pa.Table.from_pandas(pd.DataFrame(training_log)),
                        subfolder_path / "training_log.parquet",
                        compression=parquet_compression,
                    )
                if buffer_log:
                    pq.write_table(
                        pa.Table.from_pandas(pd.DataFrame(buffer_log)),
                        subfolder_path / "buffer_log.parquet",
                        compression=parquet_compression,
                    )
                return

            except Exception as exc:
                if logger is not None:
                    logger.warning("Parquet logging failed, falling back to CSV if enabled: %s", exc)

        if enable_csv_fallback:
            if training_log:
                pd.DataFrame(training_log).to_csv(
                    subfolder_path / "training_log.csv",
                    index=False,
                )
            if buffer_log:
                pd.DataFrame(buffer_log).to_csv(
                    subfolder_path / "buffer_log.csv",
                    index=False,
                )

    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to flush logs: %s", exc)