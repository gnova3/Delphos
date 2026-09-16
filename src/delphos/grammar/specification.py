"""
mdp/state.py

: author: Gabriel Nova
: date: Jun 2, 2026
: version: 0.1.0
: purpose: Handles state creation, encoding, decoding and
           backend specification generation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from .task import Task
from .catalogue import Catalogue

@dataclass(frozen=True)
class Term:
    """
    Represents one modelling term.

    Example
    -------
    Term(
        attribute_id=5,
        transform_id=2,
        taste_id=1,
        covariate_id=0
    )
    """
    attribute_id: int
    transform_id: int
    taste_id: int
    covariate_id: int


# ==========================================================
# Specification
# ==========================================================

class Specification:
    """
    Specification (state) representation used by Delphos.

    Specification tensor shape: [n_attributes, 4]
        col 0 = attribute_id
        col 1 = transform_id
        col 2 = taste_id
        col 3 = covariate_id

    Inactive attribute: 
        [attribute_id, 0, 0, 0]

    It is responsible for:
        - tensor creation
        - tensor validation
        - tensor ↔ terms conversion
        - tensor ↔ backend conversion

    State transitions belong to action.py.
    """

    ATTRIBUTE_COL = 0
    TRANSFORM_COL = 1
    TASTE_COL = 2
    COVARIATE_COL = 3
    N_COLUMNS = 4

    def __init__(self, catalogue: Catalogue, device: Optional[torch.device] = None) -> None:

        self.catalogue = catalogue
        self.device = (torch.device("cpu") if device is None else torch.device(device))

        self.attribute_ids = catalogue.attribute_ids
        self.covariate_ids = catalogue.covariate_ids
        self.transform_ids = catalogue.transform_ids
        self.taste_ids = catalogue.taste_ids
 
        self.attribute_id_to_idx = catalogue.attribute_id_to_idx
        self.covariate_id_to_idx = catalogue.covariate_id_to_idx
        self.transform_id_to_idx = catalogue.transform_id_to_idx
        self.taste_id_to_idx = catalogue.taste_id_to_idx
 
        self.n_attributes = catalogue.n_attributes

    # ======================================================
    # Empty Specification (null model)
    # ======================================================
    def empty(self) -> torch.LongTensor:
        """
        Create an empty specification (null model).

        Returns
        -------
        torch.LongTensor: Specification (n_global_attributes, 4)
        """
        specification = torch.zeros((self.n_attributes, self.N_COLUMNS), dtype=torch.long, device=self.device,)
        specification[:, self.ATTRIBUTE_COL] = torch.tensor(self.attribute_ids, dtype=torch.long, device=self.device,)
        return specification

    # ======================================================
    # Terms -> Specification
    # ======================================================

    def from_terms(self, terms: list[Term]) -> torch.LongTensor:
        """
        Create specification tensor from modelling terms.

        Args:
            terms (list[Term]): List of modelling terms.

        Returns:
            torch.LongTensor: Specification tensor.
        """
        specification = self.empty()
        seen_attributes = set()

        for term in terms:
            if term.attribute_id in seen_attributes:
                raise ValueError(f"Duplicate attribute_id {term.attribute_id}")
            
            seen_attributes.add(term.attribute_id)
            row = self.attribute_id_to_idx[term.attribute_id]
            specification[row, self.TRANSFORM_COL] = term.transform_id
            specification[row, self.TASTE_COL] = term.taste_id
            specification[row, self.COVARIATE_COL] = term.covariate_id
        return specification

    # ======================================================
    # Specificaiton -> Terms
    # ======================================================

    def to_terms(self, specification: torch.LongTensor) -> list[Term]:
        """
        Extract active modelling terms from a specification.

        Args:
            Specification (torch.LongTensor): Sequence of active attributes and their transformations, tastes and covariates.

        Returns:
            list[Term]: List of active modelling terms.

        Examples
        >>> terms = [Term(1, 1, 1, 1), Term(2, 2, 2, 2)]
        >>> specification = specification_manager.from_terms(terms)
        >>> specification_manager.to_terms(specification) == terms
        True
        """

        self.validate(specification)

        terms = []

        for row in specification:
            attribute_id = int(row[self.ATTRIBUTE_COL])
            transform_id = int(row[self.TRANSFORM_COL])
            taste_id = int(row[self.TASTE_COL])
            covariate_id = int(row[self.COVARIATE_COL])

            if (transform_id == 0 and taste_id == 0 and covariate_id == 0):
                continue

            terms.append(
                Term(
                    attribute_id=attribute_id,
                    transform_id=transform_id,
                    taste_id=taste_id,
                    covariate_id=covariate_id,
                )
            )

        return terms

    # ======================================================
    # Batch
    # ======================================================

    def batch_from_terms(self, batch_terms: list[list[Term]]) -> torch.LongTensor:
        """
        Convert multiple term lists to a tensor batch.

        Args:
            batch_terms (list[list[Term]]): List of term lists.

        Returns:
            torch.LongTensor: Batch of specifications.
        """

        return torch.stack([self.from_terms(terms) for terms in batch_terms], dim=0,)

    def batch_to_terms(self, batch_specifications: torch.LongTensor) -> list[list[Term]]:
        """
        Convert a batch of specifications into term lists.

        Args:
            batch_specifications (torch.LongTensor): Batch of specifications.

        Returns:
            list[list[Term]]: List of term lists.
        """

        return [self.to_terms(specification) for specification in batch_specifications]

    # ======================================================
    # Backend conversion (Apollo)
    # ======================================================

    def to_backend(self, specification: torch.LongTensor) -> dict:
        """
        Convert specification into backend specification.

        Args:
            specification (torch.LongTensor): Specification tensor.

        Returns:
            dict: Backend specification.
        """

        terms = self.to_terms(specification)
        
        rows = []
        attribute_ids = []
        transformation_ids = []
        taste_ids = []
        covariate_ids = []

        for term in terms:
            rows.append(
                {   "att_id": term.attribute_id,
                    "trans_id": term.transform_id,
                    "taste_id": term.taste_id,
                    "cov_id": term.covariate_id,
                }
            )
            
            attribute_ids.append(term.attribute_id)
            transformation_ids.append(term.transform_id)
            taste_ids.append(term.taste_id)
            covariate_ids.append(term.covariate_id)

        return {
            "rows": rows,
            "key": self.specification_key(specification),
            "attribute_ids": attribute_ids,
            "transformation_ids": transformation_ids,
            "taste_ids": taste_ids,
            "covariate_ids": covariate_ids,
        }

    # ======================================================
    # Specification key
    # ======================================================
    def specification_key(self, specification: torch.LongTensor) -> str:
        """
        Build a deterministic specification key.

        Args:
            specification (torch.LongTensor): Specification tensor.

        Returns:
            str: Specification key.
        """

        self.validate(specification)

        return "_".join([f"{int(row[0])}{int(row[1])}{int(row[2])}{int(row[3])}" for row in specification])

    # ======================================================
    # Validation
    # ======================================================

    def validate(self, specification: torch.LongTensor) -> None:
        """
        Validate specification consistency.

        Args:
            specification (torch.LongTensor): Specification tensor.
        """

        expected_shape = (self.n_attributes, self.N_COLUMNS)

        if tuple(specification.shape) != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, received {tuple(specification.shape)}")

        expected_attribute_ids = torch.tensor(self.attribute_ids, dtype=torch.long, device=specification.device)

        if not torch.equal(specification[:, self.ATTRIBUTE_COL], expected_attribute_ids):
            raise ValueError("Column 0 must contain catalogue attribute ids.")

    # ======================================================
    # Summary
    # ======================================================
    def summary(self) -> dict:
        """
        Return state metadata.
        """
        return {
            "n_attributes": self.n_attributes,
            "attribute_ids": self.attribute_ids,
            "covariate_ids": self.covariate_ids,
            "transform_ids": self.transform_ids,
            "taste_ids": self.taste_ids,
            "device": str(self.device),
            "shape": (
                self.n_attributes,
                self.N_COLUMNS,
            ),
        }