"""
experiments/registry.py

: author: Gabriel Nova
: date: Jun 5, 2026
: version: 0.1.0
: purpose: Experiment registry for Delphos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    def __init__(self):
        self._experiments: dict[str, dict[str, Any]] = {}

    def register(self, name: str, config: dict[str, Any], overwrite: bool = False, ) -> None:
        name = str(name)
        if name in self._experiments and not overwrite:
            raise ValueError(f"Experiment '{name}' already exists.")
        self._experiments[name] = dict(config)


    def get(self, name: str) -> dict[str, Any]:
        if name not in self._experiments:
            raise KeyError(f"Experiment '{name}' not found.")
        return self._experiments[name]

    def exists(self, name: str) -> bool:
        return name in self._experiments

    # =====================================================
    # Listing
    # =====================================================

    def list(self) -> list[str]:
        return sorted(self._experiments.keys())

    def summary(self) -> dict[str, Any]:

        return {
            "n_experiments": len(self._experiments),
            "experiments": self.list(),
        }

    def remove(self, name: str) -> None:
        if name in self._experiments:
            del self._experiments[name]

    def clear(self) -> None:
        self._experiments.clear()

    # =====================================================
    # Serialization
    # =====================================================

    def save(self, path: str | Path) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True,)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._experiments, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, ) -> "ExperimentRegistry":

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        registry = cls()
        registry._experiments = dict(data)

        return registry