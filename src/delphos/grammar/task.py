"""
mdp/task.py 

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Defines the Task class for representing discrete choice datasets.

"""


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

# ==========================================================
# Modelling terms
# ==========================================================
@dataclass(frozen=True)
class Alternative:
    """
    Represents one alternative in the choice task.

    Attributes:
        id (int): Unique identifier of the alternative.
        name (str): Name of the alternative.
        choice (int): Choice code of the alternative.
        availability (str | None): Column name of the availability of the alternative.

    Example
    -------
    Alternative(
        id=1,
        name="car",
        choice=1,
        availability="av_car"
    )
    """
    id: int
    name: str
    choice: int
    availability: str | None = None

@dataclass(frozen=True)
class Attribute:
    """
    Represents one attribute in the choice task.

    Attributes:
        id (int): Unique identifier of the attribute.
        name (str): Name of the attribute.
        alternative (dict[int, str]): Mapping of alternative IDs to attribute values.

    Example
    -------
    Attribute(
        id=1,
        name="cost",
        alternative={1: "cost_car", 2: "cost_bus", 3: "cost_train"}
    )
    """
    id: int
    name: str
    alternative: dict[int, str]

@dataclass(frozen=True)
class Covariate:
    """
    Represents one covariate in the choice task.

    Attributes:
        id (int): Unique identifier of the covariate.
        name (str): Name of the covariate.
        levels (tuple[int,...]): Levels of the covariate.
    
    Example
    -------
    age = Covariate(
            id=1,
            name="age",
            levels=(0, 1, 2, 3)
          )
    """
    id: int
    name: str
    levels: tuple[int,...]

@dataclass(frozen=True)
class Transformation:
    """
    Represents one transformation in the choice task.

    Attributes:
        id (int): Unique identifier of the transformation.
        name (str): Name of the transformation.

    Example
    -------
    log = Transformation(
            id=1,
            name="log"
          )
    """
    id: int
    name: str

@dataclass(frozen=True)
class Taste:
    """
    Represents one taste in the choice task.

    Attributes:
        id (int): Unique identifier of the taste.
        name (str): Name of the taste.

    Example
    -------
    generic = Taste(
        id=1,
        name="generic"
    )
    """
    id: int
    name: str



# ==========================================================
# TASK
# ==========================================================

@dataclass(frozen=True)
class Task:
    """
    Dataset-specific modelling task.

    Notes
    -----
    A Task is a pure description of a modelling problem.

    It contains:
    - alternatives
    - attributes
    - covariates
    - transformations
    - tastes
    - benchmark information
    """
    id: int
    name: str

    yaml_path: Path
    dataset_path: Path
    rewards_path: Path
    choice_column: str
    id_column: str
    is_panel: bool   
    
    ll_null: float
    ll_linear: float
    n_obs: int

    alternatives: tuple[Alternative,...]
    attributes: tuple[Attribute,...]
    covariates: tuple[Covariate,...]
    transformations: tuple[Transformation,...]
    tastes: tuple[Taste,...]


    # ======================================================
    # Constructors
    # ======================================================
    @classmethod
    def from_yaml(cls, id: int, yaml_path: str | Path) -> Task:
        yaml_path = Path(yaml_path).resolve()
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(id=id, config=config, yaml_path=yaml_path)

    @classmethod
    def from_dict(cls, id: int, config: dict[str, Any], yaml_path: str | Path |None = None) -> Task:

        yaml_path = (None if yaml_path is None else Path(yaml_path).resolve())
        yaml_dir = (None if yaml_path is None else yaml_path.parent)

        dataset_path = Path(config["path_choice_dataset"])
        if not dataset_path.is_absolute():
            if yaml_dir is None:
                raise ValueError("Relative dataset path requires yaml_path.")

            dataset_path = (yaml_dir / dataset_path).resolve()

        rewards_path = (yaml_dir / "rewards.sqlite").resolve()
        

        alternatives = tuple(
            Alternative(id=spec["id"], name=name, choice=spec["id"], availability=spec.get("avail"),            )
            for name, spec in config["alternatives"].items()
        )

        alternative_name_to_id = {alt.name: alt.id for alt in alternatives}

        attributes = []

        asc = Attribute(id=1, name="ASC", alternative={})
        attributes.append(asc)
        
        for name, spec in config["attributes"].items():
            mapping = {}
            for alt_name, var_name in spec["mapping"].items():
                mapping[alternative_name_to_id[alt_name]] = var_name
            attributes.append(Attribute(id=spec["id"], name=name, alternative=mapping))
        attributes = tuple(attributes)

        covariates = tuple(
            Covariate(id=spec["id"], name=name, levels=tuple(spec.get("levels", [])),)
            for name, spec in config.get("covariates", {},).items()
        )

        # Use global defaults
        linear = Transformation(id=1, name="linear")
        log = Transformation(id=2, name="log")
        box_cox =  Transformation(id=3, name="box_cox")
        transformations = (linear, log, box_cox)

        # Use global defaults
        generic = Taste(id=1, name="generic")
        specific = Taste(id=2, name="specific")
        tastes = (generic, specific) 
        
        _cls = cls(
            id=id,
            name=config.get("df_name", f"task_{id}"),
            yaml_path=yaml_path,
            dataset_path=dataset_path,
            rewards_path=rewards_path,
            choice_column=config["choice"],
            id_column=config["id"],
            is_panel=config["panel"],
            ll_null=config.get("ll_null"),
            ll_linear=config.get("ll_linear"),
            n_obs=config.get("n_obs"),
            alternatives=alternatives,    
            attributes=attributes,
            covariates=covariates,
            transformations=transformations,
            tastes=tastes,
        )

        _cls.validate()

        return _cls

    @classmethod
    def from_folder(cls, id: int, folder: str | Path) -> Task:
        folder = Path(folder)
        yaml_path = folder / "dataset.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml not found in {folder}")

        return cls.from_yaml(id=id, yaml_path=yaml_path)


    # ======================================================
    # properties
    # ======================================================
    @property
    def alternative_ids(self) -> tuple[int, ...]:
        return tuple(a.id for a in self.alternatives)

    @property
    def alternative_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.alternatives)
    
    def get_alt_by_id(self, id: int) -> Alternative:
        for alternative in self.alternatives:
            if alternative.id == id:
                return alternative
        raise KeyError(f"Unknown alternative id '{id}'")

    @property
    def attribute_ids(self) -> tuple[int, ...]:
        return tuple(a.id for a in self.attributes)

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attributes)

    def get_attr_by_id(self, id: int) -> Attribute:
        for attribute in self.attributes:
            if attribute.id == id:
                return attribute
        raise KeyError(f"Unknown attribute id '{id}'")

    @property
    def covariate_ids(self) -> tuple[int, ...]:
        return tuple(c.id for c in self.covariates if c.id is not None)

    @property
    def covariate_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.covariates)

    def get_cov_by_id(self, id: int) -> Covariate:
        for covariate in self.covariates:
            if covariate.id == id:
                return covariate
        raise KeyError(f"Unknown covariate id '{id}'")

    @property
    def transform_ids(self) -> tuple[int, ...]:
        return tuple(t.id for t in self.transformations)

    @property
    def transform_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.transformations)

    @property
    def taste_ids(self) -> tuple[int, ...]:
        return tuple(t.id for t in self.tastes)

    @property
    def taste_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tastes)

    @property
    def n_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def n_attributes(self) -> int:
        return len(self.attributes)

    @property
    def n_covariates(self) -> int:
        return len(self.covariates)

    @property
    def modelling_covariates(self):
        return tuple(cov for cov in self.covariates if cov.id is not None)

    # ======================================================
    # lookups
    # ======================================================
    def get_attribute(self, name: str) -> Attribute:
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        raise KeyError(f"Unknown attribute '{name}'")

    def get_covariate(self, name: str) -> Covariate:
        for covariate in self.covariates:
            if covariate.name == name:
                return covariate
        raise KeyError(f"Unknown covariate '{name}'")

    def get_alternative(self, name: str) -> Alternative:
        for alternative in self.alternatives:
            if alternative.name == name:
                return alternative
        raise KeyError(f"Unknown alternative '{name}'")


    # ======================================================
    # Validation
    # ======================================================
    def validate(self) -> None:
        attribute_ids = self.attribute_ids
        if len(attribute_ids) != len(set(attribute_ids)):
            raise ValueError("Duplicate attribute ids detected.")

        covariate_ids = [c.id for c in self.covariates if c.id is not None]
        if len(covariate_ids) != len(set(covariate_ids)):
            raise ValueError("Duplicate covariate ids detected.")

        alternative_ids = self.alternative_ids
        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("Duplicate alternative ids detected.")
        
        transform_ids = self.transform_ids
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("Duplicate transformation ids detected.")

        taste_ids = self.taste_ids
        if len(taste_ids) != len(set(taste_ids)):
            raise ValueError("Duplicate taste ids detected.")

    # ======================================================
    # Representation
    # ======================================================
    def __str__(self) -> str:
        return (
            f"Task("
            f"name='{self.name}', "
            f"alternatives={self.n_alternatives}, "
            f"attributes={self.n_attributes}, "
            f"covariates={self.n_covariates}"
            f")"
        )