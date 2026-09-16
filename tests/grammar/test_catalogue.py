import pytest
import delphos as dp
from delphos.grammar.catalogue import Catalogue

def test_load_global_catalogue():
    catalogue = dp.data.registry.load_global_catalogue()
    assert isinstance(catalogue, Catalogue)
    assert catalogue.n_attributes > 0
    assert catalogue.n_covariates > 0
