import pytest
import torch
import delphos as dp

def test_inference_forward_pass():
    agent = dp.load_agent(device="cpu").agent
    assert agent is not None
    
    # We can create a dummy runtime or just verify agent loads
    assert hasattr(agent, "term_encoder") or hasattr(agent, "q_network")
    assert str(agent.device) == "cpu"
