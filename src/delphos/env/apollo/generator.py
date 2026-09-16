"""
env/apollo/generator.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Apollo specification generation.
"""
from __future__ import annotations
from typing import Iterable
from delphos.grammar.task import Task
from delphos.grammar.specification import Term
from .schema import ApolloParameter
from .schema import ApolloSpecification


class ApolloGenerator:
    """
    Apollo specification generator.

    Responsibilities
    ----------------
    - Convert backend specifications into modelling terms.
    - Generate Apollo parameters.
    - Generate Apollo utility code.
    - Generate Apollo probability code.
    """

    ASC_ID = 1    
    TRANS_LINEAR = 1
    TRANS_LOG = 2
    TRANS_BOXCOX = 3
    TASTE_GENERIC = 1
    TASTE_SPECIFIC = 2
    NO_COVARIATE = 0

    def __init__(self, task: Task) -> None:
        self.task = task

    def build_terms(self, backend_specification: dict) -> list[Term]:

        terms: list[Term] = []        
        for row in backend_specification["rows"]:
            terms.append(
                Term(
                    attribute_id=row["att_id"], 
                    transform_id=row["trans_id"], 
                    taste_id=row["taste_id"], 
                    covariate_id=row["cov_id"]
                )
            )
        return terms

    def build_parameters(self, terms: list[Term],) -> list[ApolloParameter]:

        parameters: list[ApolloParameter] = []
        for term in terms:
            if term.attribute_id == self.ASC_ID:
                parameters.extend(self._build_asc_parameters(term))
            elif term.taste_id == self.TASTE_GENERIC:
                parameters.extend(self._build_generic_parameters(term))
            elif term.taste_id == self.TASTE_SPECIFIC:
                parameters.extend(self._build_specific_parameters(term))
        return self._unique_parameters(parameters)


    def _build_asc_parameters(self, term: Term,) -> list[ApolloParameter]:

        parameters: list[ApolloParameter] = []

        alternatives = list(self.task.alternatives)

        last_alt_id = alternatives[-1].id

        if term.covariate_id == self.NO_COVARIATE:
            for alt in alternatives:
                parameters.append(ApolloParameter(f"ASC_{alt.name}", 0.0, (alt.id == last_alt_id)))
            return parameters

        covariate = self.task.get_cov_by_id(term.covariate_id)
        for alt in alternatives:
            is_reference = alt.id == last_alt_id
            for level in covariate.levels:
                parameters.append(ApolloParameter(f"ASC_{alt.name}_{covariate.name}_{level}", 0.0, is_reference)) 

        return parameters

    def _build_generic_parameters(self, term: Term,) -> list[ApolloParameter]:

        parameters: list[ApolloParameter] = []

        attribute = self.task.get_attr_by_id(term.attribute_id)
        suffix = self._transform_suffix(term.transform_id)
        base_name = f"b_{attribute.name}_generic{suffix}"

        if term.transform_id == self.TRANS_BOXCOX:
            parameters.append(ApolloParameter(f"L_{attribute.name}", 1.0))

        if term.covariate_id == self.NO_COVARIATE:
            parameters.append(ApolloParameter(base_name, 0.0))
            return parameters

        covariate = self.task.get_cov_by_id(term.covariate_id)
        for level in covariate.levels:
            parameters.append(ApolloParameter(f"{base_name}_{covariate.name}_{level}", 0.0))

        return parameters


    def _build_specific_parameters(self, term: Term,) -> list[ApolloParameter]:

        parameters: list[ApolloParameter] = []

        attribute = self.task.get_attr_by_id(term.attribute_id)
        suffix = self._transform_suffix(term.transform_id)
        available_alt_ids = set(attribute.alternative.keys())

        if term.transform_id == self.TRANS_BOXCOX:
            parameters.append(ApolloParameter(f"L_{attribute.name}", 1.0))

        if term.covariate_id == self.NO_COVARIATE:
            for alt in self.task.alternatives:
                if alt.id not in available_alt_ids:
                    continue
                parameters.append(ApolloParameter(f"b_{alt.name}_{attribute.name}{suffix}", 0.0))            
            return parameters

        covariate = self.task.get_cov_by_id(term.covariate_id)
        for alt in self.task.alternatives:
            if alt.id not in available_alt_ids:
                continue
            for level in covariate.levels:
                parameters.append(ApolloParameter(f"b_{alt.name}_{attribute.name}{suffix}_{covariate.name}_{level}", 0.0))

        return parameters

    def _transform_suffix(self, transform_id: int) -> str:
        if transform_id == self.TRANS_LINEAR:
            return ""
        if transform_id == self.TRANS_LOG:
            return "_log"
        if transform_id == self.TRANS_BOXCOX:
            return "_box_cox"
        raise ValueError(f"Unknown transform id {transform_id}")

    @staticmethod
    def _unique_parameters(parameters: Iterable[ApolloParameter]) -> list[ApolloParameter]:
        unique: dict[str, ApolloParameter] = {}
        for parameter in parameters:
            unique[parameter.name] = parameter
        return list(unique.values())


    # =====================================================
    # Utility generation
    # =====================================================
    def build_utility_code(self, terms: list[Term]) -> str:
        
        utility_terms: dict[int, list[str]] = {alt.id: [] for alt in self.task.alternatives}

        for term in terms:
            if term.attribute_id == self.ASC_ID:
                self._append_asc_utility(utility_terms=utility_terms, term=term)
            elif term.taste_id == self.TASTE_GENERIC:
                self._append_generic_utility(utility_terms=utility_terms, term=term)
            else:
                self._append_specific_utility(utility_terms=utility_terms, term=term)

        lines: list[str] = []
        lines.append("V <- list()")
        for alt in self.task.alternatives:
            components = utility_terms[alt.id]
            if len(components) == 0:
                expression = "0"
            else:
                expression = " +\n      ".join(components)
            lines.append(f'V[["{alt.name}"]] <-\n      {expression}')
        return "\n\n".join(lines)


    def _append_asc_utility(self, utility_terms: dict[int, list[str]], term: Term,) -> None:

        if term.covariate_id == self.NO_COVARIATE:
            for alt in self.task.alternatives:
                utility_terms[alt.id].append(f"ASC_{alt.name}")
            return

        covariate = self.task.get_cov_by_id(term.covariate_id)
        for alt in self.task.alternatives:
            for level in covariate.levels:
                utility_terms[alt.id].append(f"ASC_{alt.name}_{covariate.name}_{level} * ({covariate.name} == {level})")

    def _append_generic_utility(self, utility_terms: dict[int, list[str]], term: Term,) -> None:

        attribute = self.task.get_attr_by_id(term.attribute_id)
        suffix = self._transform_suffix(term.transform_id)
        parameter = f"b_{attribute.name}_generic{suffix}"

        for alt_id, variable_name in attribute.alternative.items():
            variable_expr = self._variable_expression(variable_name, attribute.name, term.transform_id)
            
            if term.covariate_id == self.NO_COVARIATE:
                utility_terms[alt_id].append(f"{parameter} * {variable_expr}")
                continue

            covariate = self.task.get_cov_by_id(term.covariate_id)
            for level in covariate.levels:
                utility_terms[alt_id].append(f"{parameter}_{covariate.name}_{level} * ({covariate.name} == {level}) * {variable_expr}")



    def _append_specific_utility(self, utility_terms: dict[int, list[str]], term: Term,) -> None:

        attribute = self.task.get_attr_by_id(term.attribute_id)
        suffix = self._transform_suffix(term.transform_id)

        for alt in self.task.alternatives:
            if alt.id not in attribute.alternative:
                continue

            variable_name = attribute.alternative[alt.id]
            variable_expr = self._variable_expression(variable_name, attribute.name, term.transform_id)
            parameter = f"b_{alt.name}_{attribute.name}{suffix}"

            if term.covariate_id == self.NO_COVARIATE:
                utility_terms[alt.id].append(f"{parameter} * {variable_expr}")
                continue

            covariate = self.task.get_cov_by_id(term.covariate_id)
            for level in covariate.levels:
                utility_terms[alt.id].append(f"{parameter}_{covariate.name}_{level} * ({covariate.name} == {level}) * {variable_expr}")


    def _variable_expression(self, variable_name: str, attribute_name: str, transform_id: int,) -> str:
        if transform_id == self.TRANS_LINEAR:
            return variable_name

        if transform_id == self.TRANS_LOG:
            return f"log(1+{variable_name})"

        if transform_id == self.TRANS_BOXCOX:
            lambda_name = f"L_{attribute_name}"
            return f"(({variable_name}^{lambda_name} - 1) / {lambda_name})"

        raise ValueError(f"Unknown transform id {transform_id}")


    # =====================================================
    # Apollo probabilities
    # =====================================================

    def build_probability_code(self, utility_code: str,) -> str:
        
        panel_data = bool(self.task.is_panel)

        alternatives    = ", ".join([f"{alt.name}={alt.choice}" for alt in self.task.alternatives])
        availabilities  = ", ".join([f"{alt.name}={alt.availability}" for alt in self.task.alternatives])
        panel_code      = ("""P = apollo_panelProd(P, apollo_inputs, functionality)""" if panel_data else "")

        _apollo_probabilities = f"""
        apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {{

            apollo_attach(apollo_beta, apollo_inputs)
            on.exit(apollo_detach(apollo_beta, apollo_inputs))
            P = list()
            V = list()
            {utility_code}
            mnl_settings = list(
                alternatives = c({alternatives}),
                avail = list({availabilities}),
                choiceVar = {self.task.choice_column},
                utilities = V
            )
            P[["model"]] = apollo_mnl(mnl_settings, functionality)
            {panel_code}
            P = apollo_prepareProb(P, apollo_inputs, functionality)
            return(P)
        }}
        """
        return _apollo_probabilities


    def build_apollo_specification(self, backend_specification: dict) -> ApolloSpecification:

        terms = self.build_terms(backend_specification)
        parameters = self.build_parameters(terms)
        utility_code = self.build_utility_code(terms)
        probability_code = self.build_probability_code(utility_code)
        apollo_beta = {parameter.name: parameter.value for parameter in parameters}
        apollo_fixed = [parameter.name for parameter in parameters if parameter.fixed]

        return ApolloSpecification(
        specification_key=backend_specification["key"],
        terms=terms,
        parameters=parameters,
        utility_code=utility_code,
        probability_code=probability_code,
        apollo_beta=apollo_beta,
        apollo_fixed=apollo_fixed,
        )
