import pytest
from delphos.env.result_cache import ResultCache
import pandas as pd

def test_result_cache(tmp_path):
    cache_path = tmp_path / "test_cache"
    cache = ResultCache(str(cache_path))
    
    # Should be empty initially
    assert cache.lookup("task_1", "spec_1").empty
    
    # Add a success outcome
    df = pd.DataFrame([{"task_name": "task_1", "specification": "spec_1", "successfulEstimation": 1, "skipped": 0}])
    cache.upsert(df)
    
    # Should hit cache
    cached = cache.lookup("task_1", "spec_1")
    assert not cached.empty
    assert cached.iloc[0]["successfulEstimation"] == 1
