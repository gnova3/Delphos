import pytest
import delphos as dp
from delphos.api import DelphosModel
from delphos.grammar.task import Task
from delphos.inference.proposal import ProposalSet

def test_load_agent():
    # Load the bundled default agent
    agent = dp.load_agent(device="cpu")
    assert isinstance(agent, DelphosModel)
    assert agent.catalogue is not None

def test_list_datasets():
    # List bundled datasets
    datasets = dp.list_datasets()
    assert len(datasets) > 0

def test_load_dataset():
    # Load the first bundled dataset
    datasets = dp.list_datasets()
    task = dp.load_dataset(datasets[0].id)
    assert isinstance(task, Task)
    assert task.id == datasets[0].id

def test_propose_models():
    # Test end-to-end proposal (without Apollo estimation)
    agent = dp.load_agent(device="cpu")
    datasets = dp.list_datasets()
    task = dp.load_dataset(datasets[0].id)
    
    models = agent.propose(
        dataset=task,
        n_models=2,
        estimate=False,
        max_attempts=10
    )
    assert isinstance(models, ProposalSet)
    assert len(models.proposals) > 0
