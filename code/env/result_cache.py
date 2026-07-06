"""
env/result_cache.py

: author: Gabriel Nova
: date: Jun 2026
: version: 2.0.0
: purpose: SQLite cache for estimated model results.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Optional
import pandas as pd


RESULT_COLUMNS = [
    "task_name",
    "specification",
    "numParams",
    "numResids",
    "maximum",
    "vcHessianConditionNumber",
    "successfulEstimation",
    "LL0",
    "LLC",
    "LLout",
    "rho2_0",
    "adjRho2_0",
    "rho2_C",
    "adjRho2_C",
    "AIC",
    "BIC",
    "eigValue",
    "timeTaken",
    "nFreeParams",
    "skipped",
]

TABLE_NAME = "rewards"

@dataclass
class ResultCache:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.initialize()

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path),timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (

                    task_name TEXT NOT NULL,
                    specification TEXT NOT NULL,
                    numParams REAL,
                    numResids REAL,
                    maximum REAL,
                    vcHessianConditionNumber REAL,
                    successfulEstimation INTEGER,
                    LL0 REAL,
                    LLC REAL,
                    LLout REAL,
                    rho2_0 REAL,
                    adjRho2_0 REAL,
                    rho2_C REAL,
                    adjRho2_C REAL,
                    AIC REAL,
                    BIC REAL,
                    eigValue REAL,
                    timeTaken REAL,
                    nFreeParams REAL,
                    skipped INTEGER DEFAULT 0,
                    PRIMARY KEY (task_name, specification)
                )
                """
            )

            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                idx_{TABLE_NAME}_spec
                ON {TABLE_NAME}(specification)
                """
            )

            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                idx_{TABLE_NAME}_task
                ON {TABLE_NAME}(task_name)
                """
            )
            conn.commit()

    @staticmethod
    def empty() -> pd.DataFrame:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    def exists(self, task_name: str, specification: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM {TABLE_NAME}
                WHERE task_name = ?
                AND specification = ?
                LIMIT 1
                """,
                (str(task_name), str(specification)),
            ).fetchone()

        return row is not None

    def lookup(self, task_name: str, specification: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT *
                FROM {TABLE_NAME}
                WHERE task_name = ?
                AND specification = ?
                LIMIT 1
                """,
                conn,
                params=(str(task_name), str(specification)),
            )
        if df.empty:
            return self.empty()
        return self._normalise(df)

    def upsert(self, outcomes: pd.DataFrame,) -> None:
        if outcomes.empty:
            return

        outcomes = self._normalise(outcomes)
        records = (outcomes.where(pd.notna(outcomes), None).to_dict("records"))
        sql = f"""
        INSERT INTO {TABLE_NAME}
        (
            task_name,
            specification,
            numParams,
            numResids,
            maximum,
            vcHessianConditionNumber,
            successfulEstimation,
            LL0,
            LLC,
            LLout,
            rho2_0,
            adjRho2_0,
            rho2_C,
            adjRho2_C,
            AIC,
            BIC,
            eigValue,
            timeTaken,
            nFreeParams,
            skipped
        )
        VALUES
        (
            :task_name,
            :specification,
            :numParams,
            :numResids,
            :maximum,
            :vcHessianConditionNumber,
            :successfulEstimation,
            :LL0,
            :LLC,
            :LLout,
            :rho2_0,
            :adjRho2_0,
            :rho2_C,
            :adjRho2_C,
            :AIC,
            :BIC,
            :eigValue,
            :timeTaken,
            :nFreeParams,
            :skipped
        )
        ON CONFLICT(task_name, specification)
        DO UPDATE SET

            numParams = excluded.numParams,
            numResids = excluded.numResids,
            maximum = excluded.maximum,
            vcHessianConditionNumber = excluded.vcHessianConditionNumber,
            successfulEstimation = excluded.successfulEstimation,
            LL0 = excluded.LL0,
            LLC = excluded.LLC,
            LLout = excluded.LLout,
            rho2_0 = excluded.rho2_0,
            adjRho2_0 = excluded.adjRho2_0,
            rho2_C = excluded.rho2_C,
            adjRho2_C = excluded.adjRho2_C,
            AIC = excluded.AIC,
            BIC = excluded.BIC,
            eigValue = excluded.eigValue,
            timeTaken = excluded.timeTaken,
            nFreeParams = excluded.nFreeParams,
            skipped = excluded.skipped
        """
        with self._connect() as conn:
            conn.executemany(sql, records)
            conn.commit()

    def load(self, task_name: Optional[str] = None) -> pd.DataFrame:
        with self._connect() as conn:
            if task_name is None:
                df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}",conn)
            else:
                df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE task_name = ?",conn,params=(str(task_name),),)
        if df.empty:
            return self.empty()
        return self._normalise(df)

    @staticmethod
    def failed(task_name: str, specification: str) -> pd.DataFrame:
        row = {col: pd.NA for col in RESULT_COLUMNS}
        row["task_name"] = task_name
        row["specification"] = specification
        row["successfulEstimation"] = 0
        row["skipped"] = 0
        return pd.DataFrame([row])

    def _normalise(self, df: pd.DataFrame,) -> pd.DataFrame:
        df = df.copy()
        for col in RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df["successfulEstimation"] = df["successfulEstimation"].fillna(0).astype(int)
        df["skipped"] = df["skipped"].fillna(0).astype(int)
        return df[RESULT_COLUMNS]

    @staticmethod
    def skipped(task_name: str, specification: str, n_free_parameters: int,) -> pd.DataFrame:
        row = {col: pd.NA for col in RESULT_COLUMNS}
        row["task_name"] = task_name
        row["specification"] = specification
        row["successfulEstimation"] = 0
        row["skipped"] = 1
        row["numParams"] = n_free_parameters
        row["nFreeParams"] = n_free_parameters
        return pd.DataFrame([row])
