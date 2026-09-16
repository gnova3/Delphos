import pytest
from pathlib import Path
from delphos.grammar.task import Task
import delphos as dp

def test_load_bundled_task():
    datasets = dp.list_datasets()
    assert len(datasets) > 0
    task = dp.load_dataset(datasets[0].id)
    assert isinstance(task, Task)
    assert task.name is not None
    assert task.choice_column is not None
    assert len(task.alternatives) > 0
