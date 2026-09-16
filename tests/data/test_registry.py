import pytest
import delphos as dp
from delphos.data.registry import list_datasets, load_dataset

def test_list_and_load_datasets():
    datasets = list_datasets()
    assert len(datasets) > 0
    # Test loading by ID
    task1 = load_dataset(datasets[0].id)
    assert task1.id == datasets[0].id
    
    # Test loading by Name
    task2 = load_dataset(datasets[0].name)
    assert task2.id == datasets[0].id

def test_load_dataset_invalid():
    with pytest.raises(KeyError):
        load_dataset(999999)
    with pytest.raises(KeyError):
        load_dataset("UnknownDatasetThatDoesNotExist")
