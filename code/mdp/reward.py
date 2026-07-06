"""
mdp/reward.py
: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Reward functions used by Delphos.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .task import Task



def is_valid_estimation(modelling_outcome: pd.DataFrame) -> bool:
    if modelling_outcome is None:
        return False

    if modelling_outcome.empty:
        return False

    if "LLout" not in modelling_outcome.columns:
        return False

    if "skipped" in modelling_outcome.columns:
        if int(modelling_outcome["skipped"].iloc[0]) == 1:
            return False

    if "successfulEstimation" in modelling_outcome.columns:
        success = modelling_outcome["successfulEstimation"].iloc[0]
        if pd.isna(success):
            return False
        if not bool(success):
            return False
    
    ll_out = modelling_outcome["LLout"].iloc[0]
    if pd.isna(ll_out):
        return False
    return True

def reward_function(task: Task, modelling_outcome: pd.DataFrame) -> float:
    """Calculate the reward associated with a model estimation outcome.

    The reward is computed from improvements in log-likelihood and
    information criteria relative to baseline specifications.

    Args:
        task (Task): Task environment configuration containing baseline metrics and dataset information.
        modelling_outcome (pd.DataFrame): DataFrame containing model estimation outputs and diagnostics.

    Returns:
        float: Reward value associated with the estimated specification.
    """
    if not is_valid_estimation(modelling_outcome):
        return -1.0

    n_obs = int(task.n_obs)
    ll_null = float(task.ll_null)
    ll_linear = float(task.ll_linear)
    ll_out = float(modelling_outcome["LLout"].iloc[0])

    k_null      = 0
    k_linear    = task.n_attributes
    k_out       = modelling_outcome["nFreeParams"].iloc[0]    
   
    bic_null    = k_null*np.log(n_obs) - 2* ll_null
    bic_linear  = k_linear*np.log(n_obs) - 2* ll_linear
    bic_out     = k_out*np.log(n_obs)  - 2 * ll_out    
    

    # Loglikelihood improvement agains linear additive model
    r_ll_linear = (ll_out - ll_linear) / n_obs
    
    # Loglikelihood improvement agains null model
    r_ll_null = (ll_out - ll_null) / n_obs
    
    # BIC improvement agains linear additive model
    r_bic_linear = (bic_linear - bic_out) / n_obs

    # BIC improvement agains null model
    r_bic_null =   (bic_null - bic_out) / n_obs

    # BIC improvement against linear additive model normalised by null model 
    r_bic_null_linear = (bic_linear - bic_out) / (bic_null - bic_out + 1e-8)

    return float(np.tanh(r_ll_null))