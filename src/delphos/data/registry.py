from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from delphos.grammar.catalogue import Catalogue
from delphos.grammar.task import Task


@dataclass(frozen=True)
class DatasetInfo:
    id: int
    folder: str
    name: str
    path: Path


import importlib.resources

def dataset_root() -> Path:
    return Path(str(importlib.resources.files("delphos.data.bundled.datasets")))

def checkpoints_root() -> Path:
    return Path(str(importlib.resources.files("delphos.data.bundled.checkpoints")))


def default_checkpoint_path() -> Path:
    path = checkpoints_root() / "full_agent_task_10_seed_123" / "latest_checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Default Delphos checkpoint not found: {path}")
    return path


def list_datasets() -> list[DatasetInfo]:
    infos: list[DatasetInfo] = []
    yaml_paths = sorted(
        dataset_root().glob("dataset_*/dataset.yaml"),
        key=lambda path: _folder_id(path.parent.name),
    )
    for yaml_path in yaml_paths:
        folder = yaml_path.parent.name
        dataset_id = _folder_id(folder)
        task = Task.from_yaml(id=dataset_id, yaml_path=yaml_path)
        infos.append(
            DatasetInfo(
                id=dataset_id,
                folder=folder,
                name=task.name,
                path=yaml_path.parent,
            )
        )
    return infos


def load_dataset(dataset: str | int) -> Task:
    infos = list_datasets()
    if isinstance(dataset, int):
        for info in infos:
            if info.id == dataset:
                return Task.from_folder(id=info.id, folder=info.path)
        raise KeyError(f"Unknown bundled dataset id: {dataset}")

    query = str(dataset).strip().lower()
    for info in infos:
        if query in {info.name.lower(), info.folder.lower(), str(info.id)}:
            return Task.from_folder(id=info.id, folder=info.path)

    names = ", ".join(info.name for info in infos)
    raise KeyError(f"Unknown bundled dataset '{dataset}'. Available: {names}")


def load_global_catalogue(tasks: list[Task] | None = None) -> Catalogue:
    if tasks is None:
        tasks = [Task.from_folder(id=info.id, folder=info.path) for info in list_datasets()]
    return Catalogue.from_tasks(tasks)


def _folder_id(folder: str) -> int:
    if folder.startswith("dataset_"):
        suffix = folder.removeprefix("dataset_")
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Dataset folder must be named dataset_<id>, got {folder!r}")
