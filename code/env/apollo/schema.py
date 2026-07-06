"""
env/apollo/schema.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Apollo specification objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from mdp.state import Term


# ==========================================================
# Apollo Parameter
# ==========================================================

@dataclass(frozen=True)
class ApolloParameter:
    """
    Apollo parameter definition.

    Examples
    --------
    ApolloParameter(name="b_time", initial_value=0.0, fixed=False)
    ApolloParameter(name="lambda_time", initial_value=1.0, fixed=False)
    """

    name: str
    value: float = 0.0
    fixed: bool = False


# ==========================================================
# Apollo Specification
# ==========================================================

@dataclass
class ApolloSpecification:
    """
    Complete Apollo model specification.

    Generated from a Delphos backend specification.

    Attributes
    ----------
    specification_key: Unique Delphos specification identifier.
    terms: Active modelling terms.
    parameters: Apollo parameters.
    utility_code: Generated utility expressions.
    probability_code: Generated Apollo probability function.
    """

    specification_key: str
    terms: list[Term]
    parameters: list[ApolloParameter]
    utility_code: str
    probability_code: str
    apollo_beta: dict[str, float]
    apollo_fixed: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def parameter_names(self) -> list[str]:
        """
        Return parameter names.
        """
        return [parameter.name for parameter in self.parameters]

    @property
    def parameter_values(self) -> list[float]:
        """
        Return initial values.
        """
        return [parameter.initial_value for parameter in self.parameters]

    @property
    def fixed_parameters(self) -> list[str]:
        """
        Return fixed parameter names.
        """
        return [parameter.name for parameter in self.parameters if parameter.fixed]

    @property
    def estimated_parameters(self) -> list[str]:
        """
        Return estimated parameter names.
        """
        return [parameter.name for parameter in self.parameters if not parameter.fixed]

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def summary(self) -> dict:
        """
        Return Apollo specification summary.
        """

        return {
            "specification_key": self.specification_key,
            "n_terms": len(self.terms),
            "n_parameters": self.n_parameters,
            "n_fixed": len(self.fixed_parameters),
            "n_estimated": len(self.estimated_parameters),
        }