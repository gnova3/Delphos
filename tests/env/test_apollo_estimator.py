import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import delphos as dp
from delphos.env.apollo.estimator import run_apollo_estimation
from delphos.env.apollo.schema import ApolloSpecification

@patch("subprocess.run")
@patch("shutil.which")
def test_run_apollo_estimation_mocked(mock_which, mock_run, tmp_path):
    mock_which.return_value = "/usr/bin/Rscript"
    mock_run.return_value = MagicMock(returncode=0)
    
    datasets = dp.list_datasets()
    task = dp.load_dataset(datasets[0].id)
    
    spec = ApolloSpecification(
        specification_key="mock_model",
        terms=[],
        parameters=[],
        utility_code=[],
        apollo_beta={"b_test": 0.0},
        apollo_fixed=[],
        probability_code="mock_code"
    )
    
    # Pre-create the expected output CSV so pandas can read it
    summary_path = tmp_path / "mock_model_summary.csv"
    pd.DataFrame({"AIC": [100.0], "LL0": [-200.0]}).to_csv(summary_path, index=False)
    
    summary = run_apollo_estimation(
        task=task,
        apollo_specification=spec,
        output_directory=tmp_path,
        info=False,
        save=False,
    )
    
    assert "skipped" in summary.columns
    assert summary["AIC"].iloc[0] == 100.0
    
    # Assert subprocess was called correctly
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0][0] == "Rscript"
