import pytest
from delphos.env.apollo.schema import ApolloSpecification

def test_apollo_specification_code_generation():
    spec = ApolloSpecification(
        specification_key="test_spec",
        terms=[],
        parameters=[],
        utility_code=[],
        apollo_beta={"b_time": 0.0, "b_cost": 0.0},
        apollo_fixed=["b_cost"],
        probability_code="V <- list()\nmnl_settings <- list(V=V)"
    )
    assert spec.specification_key == "test_spec"
    assert spec.apollo_beta["b_time"] == 0.0
    assert "b_cost" in spec.apollo_fixed
