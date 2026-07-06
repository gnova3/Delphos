from __future__ import annotations

from pathlib import Path
import sys

import torch

# Temporary helper if the project is not yet installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdp.dataset_schema import (
    DatasetSchema,
    Alternative,
    Attribute,
    Covariate,
    Transformation,
    Taste,
    Asc,
)
from mdp.task import Task



# =============================================================================
# Shared helpers
# =============================================================================
ASC = Asc(id=1, name="asc")

LINEAR  = Transformation(id=1, name="Linear", description="Linear transformation")
LOG     = Transformation(id=2, name="Log", description="Logarithmic transformation")
BOX_COX = Transformation(id=3, name="Box-Cox", description="Box-Cox transformation")
DEFAULT_TRANSFORMATIONS = (LINEAR, LOG, BOX_COX)

GENERIC = Taste(id=1, name="generic", description="Generic taste across alternatives")
SPECIFIC = Taste(id=2, name="specific", description="Alternative-specific taste per alternative")
DEFAULT_TASTES = (GENERIC, SPECIFIC)

GLOBAL_ATTRIBUTE_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
GLOBAL_COVARIATE_IDS = (1, 2, 3, 4, 5, 6, 7)


def make_alternative(alt_id: int) -> Alternative:
    """Create a generic alternative definition.

    Args:
        alt_id (int): The integer ID of the alternative.

    Returns:
        Alternative: The initialized alternative object.
    """
    return Alternative(
        id=alt_id,
        name=f"alt_{alt_id}",
        description=f"Alternative {alt_id}",
        choice_code=alt_id,
    )

def make_mask(full_ids: tuple[int, ...], active_ids: tuple[int, ...]) -> torch.BoolTensor:
    """Build a boolean mask over a full ordered id set.

    Args:
        full_ids (tuple[int, ...]): The full sequence of IDs.
        active_ids (tuple[int, ...]): The subset of active IDs.

    Returns:
        torch.BoolTensor: A boolean tensor mask of the same length as full_ids.
    """
    active_set = set(active_ids)
    return torch.tensor([item_id in active_set for item_id in full_ids], dtype=torch.bool)





# =============================================================================
# Dataset 1: ApolloModeChoice
# =============================================================================
name_v1 = "ApolloModeChoice"
path_choice_dataset_v1 = "dataset/dataset_1/2019_apollo_modechoice_formatted.csv"
path_rewards_dataset_v1 = "dataset/dataset_1"
task_v1 = Task.from_yaml(0, PROJECT_ROOT.parent / "dataset/dataset_1/dataset.yaml")

attribute_names_v1 = ("asc", "time", "cost", "access", "service")
covariate_names_v1 = ("female", "income", "business")
covariate_levels_v1 = {
    "female": (0, 1),
    "income": (1, 2, 3, 4),
    "business": (0, 1),
}

attribute_ids_v1 = (1, 2, 3, 4, 6)
covariate_ids_v1 = (1, 2, 6)
attribute_mask_v1 = make_mask(GLOBAL_ATTRIBUTE_IDS, attribute_ids_v1)
covariate_mask_v1 = make_mask(GLOBAL_COVARIATE_IDS, covariate_ids_v1)

alt_1_v1 = make_alternative(1)
alt_2_v1 = make_alternative(2)
alt_3_v1 = make_alternative(3)
alt_4_v1 = make_alternative(4)

att_1_v1 = Attribute(id=2, name="time", alternative_ids=(1, 2, 3, 4),)
att_2_v1 = Attribute(id=3, name="cost", alternative_ids=(1, 2, 3, 4),)
att_3_v1 = Attribute(id=4, name="access", alternative_ids=(2, 3, 4),)
att_4_v1 = Attribute(id=6, name="service", alternative_ids=(3, 4),)

cov_1_v1 = Covariate(id=1, name="female", levels=(0, 1),)
cov_2_v1 = Covariate(id=2, name="income", levels=(1, 2, 3, 4),)
cov_3_v1 = Covariate(id=6, name="business", levels=(0, 1),)

rules_v1: tuple[str, ...] = ()
is_panel_v1 = True

LL_null_v1 = float(-7550.96419769869)# check this
LL_linear_v1 = float(-6250.7023229366)# check this
N_obs_v1 = int(6400)# check this

dataset_schema_v1 = DatasetSchema(
    name=name_v1,
    asc=ASC,
    alternatives=(alt_1_v1, alt_2_v1, alt_3_v1, alt_4_v1),
    attributes=(att_1_v1, att_2_v1, att_3_v1, att_4_v1),
    transformations=DEFAULT_TRANSFORMATIONS,
    tastes=DEFAULT_TASTES,
    covariates=(cov_1_v1, cov_2_v1, cov_3_v1),
    domain_rules=rules_v1,
    is_panel=is_panel_v1,
)


# =============================================================================
# Dataset 2: SwissmetroRouteChoice
# =============================================================================
name_v2 = "SwissmetroRouteChoice"
path_choice_dataset_v2 = "dataset/dataset_2/2018_apollo_routechoice_formatted.csv"
path_rewards_dataset_v2 = "dataset/dataset_2"
task_v2 = Task.from_yaml(1, PROJECT_ROOT.parent / "dataset/dataset_2/dataset.yaml")

attribute_names_v2 = ("asc", "time", "cost", "headway", "interchanges")
covariate_names_v2 = ("hh_inc_abs", "purpose", "car_availability", "business")
covariate_levels_v2 = {
    "hh_inc_abs": (0, 1, 2, 3, 4),
    "purpose": (1, 2, 3, 4),
    "car_availability": (0, 1),
    "business": (0, 1),
}

attribute_ids_v2 = (1, 2, 3, 4, 5)
covariate_ids_v2 = (2, 4, 5, 6)
attribute_mask_v2 = make_mask(GLOBAL_ATTRIBUTE_IDS, attribute_ids_v2)
covariate_mask_v2 = make_mask(GLOBAL_COVARIATE_IDS, covariate_ids_v2)

alt_1_v2 = make_alternative(1)
alt_2_v2 = make_alternative(2)

att_1_v2 = Attribute(id=2, name="time", alternative_ids=(1, 2),)
att_2_v2 = Attribute(id=3, name="cost", alternative_ids=(1, 2),)
att_3_v2 = Attribute(id=4, name="headway", alternative_ids=(1, 2), )
att_4_v2 = Attribute(id=5, name="interchanges",alternative_ids=(1, 2),)

cov_1_v2 = Covariate(id=2, name="hh_inc_abs", levels=(0, 1, 2, 3, 4),)
cov_2_v2 = Covariate(id=4, name="purpose", levels=(1, 2, 3, 4),)
cov_3_v2 = Covariate(id=5, name="car_availability", levels=(0, 1),)
cov_4_v2 = Covariate(id=6, name="business", levels=(0, 1),)

rules_v2: tuple[str, ...] = ()
is_panel_v2 = True

LL_null_v2 = float(-1933.88063376224)# check this
LL_linear_v2 = float(-1337.88755786563)# check this
N_obs_v2 = int(2790)# check this

dataset_schema_v2 = DatasetSchema(
    name=name_v2,
    asc=ASC,
    alternatives=(alt_1_v2, alt_2_v2),
    attributes=(att_1_v2, att_2_v2, att_3_v2, att_4_v2),
    transformations=DEFAULT_TRANSFORMATIONS,
    tastes=DEFAULT_TASTES,
    covariates=(cov_1_v2, cov_2_v2, cov_3_v2, cov_4_v2),
    domain_rules=rules_v2,
    is_panel=is_panel_v2,
)


# =============================================================================
# Dataset 3: Decisions
# =============================================================================
name_v3 = "Decisions"
path_choice_dataset_v3 = "dataset/dataset_3/2020_decisions_formatted.csv"
path_rewards_dataset_v3 = "dataset/dataset_3"
task_v3 = Task.from_yaml(2, PROJECT_ROOT.parent / "dataset/dataset_3/dataset.yaml")

attribute_names_v3 = ("asc", "time", "cost", "access_time")
covariate_names_v3 = ("female", "income_perso", "age", "purpose", "n_car", "education")
covariate_levels_v3 = {
    "female": (0, 1),
    "income_perso": (1, 2, 3, 4, 5, 6),
    "age": (1,2, 3, 4, 5),
    "purpose": (1, 2, 3, 4),
    "n_car": (0, 1, 2),
    "education": (1, 2, 3, 4),
}
attribute_ids_v3 = (1, 2, 3, 4)
covariate_ids_v3 = (1, 2, 3, 4, 5, 7)
attribute_mask_v3 = make_mask(GLOBAL_ATTRIBUTE_IDS, attribute_ids_v3)
covariate_mask_v3 = make_mask(GLOBAL_COVARIATE_IDS, covariate_ids_v3)

alts_v3 = tuple(make_alternative(i) for i in range(1, 7))

att_1_v3 = Attribute(id=2, name="time", alternative_ids=(1, 2, 3, 4, 5, 6), )
att_2_v3 = Attribute(id=3, name="cost", alternative_ids=(1, 2, 3, 4), )
att_3_v3 = Attribute(id=4, name="access_time", alternative_ids=(2, 3),)

cov_1_v3 = Covariate(id=1, name="female", levels=(0, 1),)
cov_2_v3 = Covariate(id=2, name="income_perso", levels=(1, 2, 3, 4, 5, 6),)
cov_3_v3 = Covariate(id=3, name="age", levels=(1,2, 3, 4, 5),)
cov_4_v3 = Covariate(id=4, name="purpose", levels=(1, 2, 3, 4),)
cov_5_v3 = Covariate(id=5, name="n_car", levels=(0, 1, 2),)
cov_6_v3 = Covariate(id=7, name="education", levels=(1, 2, 3, 4),)

rules_v3: tuple[str, ...] = ()
is_panel_v3 = False

LL_null_v3 = float(-11965.3763878499)# check this
LL_linear_v3 = float(-4783.21430651218)# check this
N_obs_v3 = int(10019)# check this

dataset_schema_v3 = DatasetSchema(
    name=name_v3,
    asc=ASC,
    alternatives=alts_v3,
    attributes=(att_1_v3, att_2_v3, att_3_v3),
    transformations=DEFAULT_TRANSFORMATIONS,
    tastes=DEFAULT_TASTES,
    covariates=(cov_1_v3, cov_2_v3, cov_3_v3, cov_4_v3, cov_5_v3, cov_6_v3),
    domain_rules=rules_v3,
    is_panel=is_panel_v3,
)


# =============================================================================
# Dataset 4: Swissmetro
# =============================================================================
name_v4 = "Swissmetro"
path_choice_dataset_v4 = "dataset/dataset_4/2001_swissmetro_formatted.csv"
path_rewards_dataset_v4 = "dataset/dataset_4"
task_v4 = Task.from_yaml(3, PROJECT_ROOT.parent / "dataset/dataset_4/dataset.yaml")

attribute_names_v4 = ("asc", "time", "cost", "headway_time", "seat_availability")
covariate_names_v4 = ("male", "income", "age", "purpose", "first")
covariate_levels_v4 = {
    "male": (0, 1), 
    "income": (1, 2, 3, 4), 
    "age": (1, 2, 3, 4, 5),
    "purpose": (1, 2),
    "first": (0, 1),}

attribute_ids_v4 = (1, 2, 3, 4, 6)
covariate_ids_v4 = (1, 2, 3, 4, 6)
attribute_mask_v4 = make_mask(GLOBAL_ATTRIBUTE_IDS, attribute_ids_v4)
covariate_mask_v4 = make_mask(GLOBAL_COVARIATE_IDS, covariate_ids_v4)

alt_1_v4 = make_alternative(1)
alt_2_v4 = make_alternative(2)
alt_3_v4 = make_alternative(3)

att_1_v4 = Attribute(id=2, name="time", alternative_ids=(1, 2, 3),)
att_2_v4 = Attribute(id=3, name="cost", alternative_ids=(1, 2),)
att_3_v4 = Attribute(id=4, name="headway_time", alternative_ids=(1, 2),)
att_4_v4 = Attribute(id=6, name="seat_availability", alternative_ids=(2,),)

cov_1_v4 = Covariate(id=1, name="male", levels=(0, 1))
cov_2_v4 = Covariate(id=2, name="income", levels=(1, 2, 3, 4))
cov_3_v4 = Covariate(id=3, name="age", levels=(1, 2, 3, 4, 5))
cov_4_v4 = Covariate(id=4, name="purpose", levels=(1,2))
cov_5_v4 = Covariate(id=6, name="first", levels=(0, 1))

rules_v4: tuple[str, ...] = ()
is_panel_v4 = True

LL_null_v4 = float(-5548.28178432467)
LL_linear_v4 = float(-4346.73503300355)
N_obs_v4 = int(5409)

dataset_schema_v4 = DatasetSchema(
    name=name_v4,
    asc=ASC,
    alternatives=(alt_1_v4, alt_2_v4, alt_3_v4),
    attributes=(att_1_v4, att_2_v4, att_3_v4, att_4_v4),
    transformations=DEFAULT_TRANSFORMATIONS,
    tastes=DEFAULT_TASTES,
    covariates=(cov_1_v4, cov_2_v4, cov_3_v4, cov_4_v4, cov_5_v4),
    domain_rules=rules_v4,
    is_panel=is_panel_v4,
)