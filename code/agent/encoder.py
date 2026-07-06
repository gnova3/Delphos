from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.init as init


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def count_parameters(module: nn.Module) -> int:
    """Count the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _init_linear_weights(module: nn.Module) -> None:
    """Kaiming initialization for linear layers."""
    if isinstance(module, nn.Linear):
        init.kaiming_uniform_(
            module.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="relu",
        )
        if module.bias is not None:
            init.zeros_(module.bias)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
@dataclass
class ZStateConfig:
    """Configuration for Delphos state encoders.

    The configuration defines the dimensionality of:
    - the modelling grammar 
    - the neural architecture 
    used to transform model specifications into fixed-length latent representations.

    Attributes:
        K: Number of attributes [int].
        T: Number of transformations [int].
        G: Number of taste types [int].
        C: Number of covariates [int].
        d_att: Embedding size for attributes [int]. Defaults to 16.
        d_tr: Embedding size for transformations [int]. Defaults to 8.
        d_taste: Embedding size for tastes [int]. Defaults to 8.
        d_cov: Embedding size for covariates [int]. Defaults to 16.
        d_term: Output size of the term encoder [int]. Defaults to 64.
        d_state: Final state embedding size if head_flag=True [int]. Defaults to 128.
        context_dim: Dimension of the external dataset-context vector [int]. Defaults to 0.
        head_flag: Whether to use an explicit head projection [bool]. Defaults to False.
        pooling: Pooling strategy, e.g., 'mean' or 'sum' [str]. Defaults to "mean".
        attention_heads: Number of attention heads for interaction encoder [int]. Defaults to 4.
        attention_layers: Number of attention layers [int]. Defaults to 1.
        attention_dropout: Dropout rate for attention [float]. Defaults to 0.0.
    """
    K: int
    T: int
    G: int
    C: int

    d_att: int = 16
    d_tr: int = 8
    d_taste: int = 8
    d_cov: int = 16
    d_term: int = 64
    d_state: int = 128

    context_dim: int = 0
    head_flag: bool = False
    pooling: str = "mean"

    attention_heads: int = 4
    attention_layers: int = 1
    attention_dropout: float = 0.0


# -------------------------------------------------------------------------
# Term encoder
# -------------------------------------------------------------------------
class TermEncoder(nn.Module):
    """
    Embedding-based term encoder.

    The encoder transforms each model term into a dense vector
    representation using learned embeddings and a shared MLP.

    The resulting embedding captures relationships between modelling
    decisions while remaining independent of specification order.

    The encoder:
        1. looks up embeddings for each component,
        2. concatenates them,
        3. passes the result through a shared MLP,
        4. returns one embedding per term.
    """

    def __init__(self, cfg: ZStateConfig, hidden_layers: Sequence[int] = (128, 64),) -> None:
        super().__init__()
        self.cfg = cfg

        self.E_att      = nn.Embedding(num_embeddings=cfg.K + 1, embedding_dim=cfg.d_att)
        self.E_tr       = nn.Embedding(num_embeddings=cfg.T + 1, embedding_dim=cfg.d_tr)
        self.E_taste    = nn.Embedding(num_embeddings=cfg.G + 1, embedding_dim=cfg.d_taste)
        self.E_cov      = nn.Embedding(num_embeddings=cfg.C + 1, embedding_dim=cfg.d_cov)

        d_in = cfg.d_att + cfg.d_tr + cfg.d_taste + cfg.d_cov

        layers: list[nn.Module] = []
        last_size = d_in
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size

        layers.append(nn.Linear(last_size, cfg.d_term))
        self.mlp = nn.Sequential(*layers)

        self.apply(_init_linear_weights)

    def _check_id_range(self, x: torch.LongTensor, max_id: int, name: str) -> None:
        if x.numel() == 0:
            return
        if x.min().item() < 0 or x.max().item() > max_id:
            raise ValueError(
                f"{name} ids out of range. Got min={x.min().item()}, max={x.max().item()}, "
                f"expected within [0, {max_id}]."
            )

    def forward(self, k_ids: torch.LongTensor, t_ids: torch.LongTensor, g_ids: torch.LongTensor, i_ids: torch.LongTensor,) -> torch.Tensor:
        """Forward pass for the term encoder.

        Args:
            k_ids (torch.LongTensor): Tensors of shape [L] containing IDs for attributes.
            t_ids (torch.LongTensor): Tensors of shape [L] containing IDs for transformations.
            g_ids (torch.LongTensor): Tensors of shape [L] containing IDs for tastes.
            i_ids (torch.LongTensor): Tensors of shape [L] containing IDs for covariates.

        Returns:
            torch.Tensor: Tensor of shape [L, d_term] with one embedding per term.

        Raises:
            ValueError: If inputs are not 1D or differ in length.
        """
        if not (k_ids.dim() == t_ids.dim() == g_ids.dim() == i_ids.dim() == 1):
            raise ValueError(
                f"All term id tensors must be 1D. Got shapes: "
                f"{tuple(k_ids.shape)}, {tuple(t_ids.shape)}, {tuple(g_ids.shape)}, {tuple(i_ids.shape)}"
            )
        
        L = k_ids.size(0)
        if not (t_ids.size(0) == g_ids.size(0) == i_ids.size(0) == L):
            raise ValueError(
                f"All term id tensors must have same length. Got lengths: "
                f"{L}, {t_ids.size(0)}, {g_ids.size(0)}, {i_ids.size(0)}"
            )
        
        self._check_id_range(k_ids, self.cfg.K, "attribute")
        self._check_id_range(t_ids, self.cfg.T, "transform")
        self._check_id_range(g_ids, self.cfg.G, "taste")
        self._check_id_range(i_ids, self.cfg.C, "covariate")
        
        e_att   = self.E_att(k_ids)
        e_tr    = self.E_tr(t_ids)
        e_taste = self.E_taste(g_ids)
        e_cov   = self.E_cov(i_ids)

        x       = torch.cat([e_att, e_tr, e_taste, e_cov], dim=-1)
        h_terms = self.mlp(x)

        return h_terms


# -------------------------------------------------------------------------
# Shared post-pooling utilities
# -------------------------------------------------------------------------
class _BaseSpecificationEncoder(nn.Module):  
    """
    Base class for specification-level state encoders.

    A set of modelling terms that must be transformed into a fixed-length state representation.

    This class provides shared functionality for:
        - pooling term embeddings,
        - incorporating task-context information,
        - applying optional projection heads,
        - producing final state representations.

    Derived classes define how interactions between modelling terms
    are represented before pooling.
    """  

    def __init__(self, cfg: ZStateConfig, term_encoder: nn.Module) -> None:
        super().__init__()
        assert cfg.pooling in ("mean", "sum"), "pooling must be 'mean' or 'sum'"

        self.cfg = cfg
        self.term_encoder = term_encoder
        self.pooling = cfg.pooling
        self.head_flag = cfg.head_flag

        d_in = cfg.d_term + cfg.context_dim
        self.head = nn.Sequential(nn.Linear(d_in, cfg.d_state), nn.ReLU(),)

        self.empty_state = nn.Parameter(torch.zeros(cfg.d_term))
        self.head.apply(_init_linear_weights)

    def _pool_terms(self, h_terms: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h_terms.mean(dim=0)
        return h_terms.sum(dim=0)

    def _append_context(self, z_spec: torch.Tensor, context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.context_dim == 0:
            return z_spec

        assert context_vector is not None, "context_vector must be provided when context_dim > 0"

        if context_vector.dim() == 2 and context_vector.size(0) == 1:
            context_vector = context_vector.squeeze(0)

        if context_vector.dim() != 1:
            raise ValueError(f"context_vector must be 1D after squeeze, got shape {tuple(context_vector.shape)}")

        if context_vector.numel() != self.cfg.context_dim:
            raise ValueError(
                f"context_vector has size {context_vector.numel()}, expected {self.cfg.context_dim}"
            )
        
        context_vector = context_vector.to(device=z_spec.device, dtype=z_spec.dtype)
        return torch.cat([z_spec, context_vector], dim=-1)

    def _postprocess(self, z_in: torch.Tensor) -> torch.Tensor:
        if self.head_flag:
            return self.head(z_in)
        return z_in

    def _encode_terms_to_spec( self, k_ids: torch.LongTensor, t_ids: torch.LongTensor, g_ids: torch.LongTensor, i_ids: torch.LongTensor, ) -> torch.Tensor:
        raise NotImplementedError
    

    def forward(self, k_ids: torch.LongTensor, t_ids: torch.LongTensor, g_ids: torch.LongTensor, i_ids: torch.LongTensor, context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        z_spec = self._encode_terms_to_spec(k_ids, t_ids, g_ids, i_ids)
        z_in = self._append_context(z_spec, context_vector)
        z_state = self._postprocess(z_in)
        return z_state
    
    def summary(self) -> dict:
        return {
            "K": self.cfg.K,
            "T": self.cfg.T,
            "G": self.cfg.G,
            "C": self.cfg.C,
            "d_att": self.cfg.d_att,
            "d_tr": self.cfg.d_tr,
            "d_taste": self.cfg.d_taste,
            "d_cov": self.cfg.d_cov,
            "d_term": self.cfg.d_term,
            "d_state": self.cfg.d_state,
            "context_dim": self.cfg.context_dim,
            "head_flag": self.cfg.head_flag,
            "pooling": self.cfg.pooling,
            "n_parameters": count_parameters(self),
        }



# -------------------------------------------------------------------------
# Baseline Deep Sets encoder
# -------------------------------------------------------------------------
class DeepSetEncoder(_BaseSpecificationEncoder):
    """Permutation-invariant specification encoder.

    Each modelling term is encoded independently and then
    aggregated using a pooling operation.
    """

    def __init__(self, cfg: ZStateConfig, term_encoder: nn.Module) -> None:
        super().__init__(cfg, term_encoder)

    def _encode_terms_to_spec(self, k_ids: torch.LongTensor, t_ids: torch.LongTensor, g_ids: torch.LongTensor, i_ids: torch.LongTensor, ) -> torch.Tensor:
        L = k_ids.size(0)

        if L == 0:
            return self.empty_state

        h_terms = self.term_encoder(k_ids, t_ids, g_ids, i_ids)
        z_spec = self._pool_terms(h_terms)
        return z_spec


# -------------------------------------------------------------------------
# Interaction-aware blocks
# -------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """
    Self-attention block over term embeddings.

    No positional encoding is used, so the block remains permutation-equivariant.
    Pooling afterwards yields a permutation-invariant state representation.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, ff_multiplier: int = 2, ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,)

        ff_dim = ff_multiplier * d_model

        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model),)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Tensor of shape [B, L, d_model].

        Returns
        -------
        torch.Tensor
            Tensor of shape [B, L, d_model].
        """
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm_1(x + self.dropout(attn_out))

        ff_out = self.ffn(x)
        x = self.norm_2(x + self.dropout(ff_out))
        return x


# -------------------------------------------------------------------------
# Interaction-aware encoder
# -------------------------------------------------------------------------
class InteractionAwareEncoder(_BaseSpecificationEncoder):
    """
    Interaction-aware specification encoder.

    Pipeline:
      1. Encode each term independently with TermEncoder.
      2. Pass term embeddings through one or more self-attention blocks.
      3. Pool the updated term embeddings.
      4. Append the explicit dataset-context vector.
      5. Optionally apply a final head.

    This preserves permutation invariance at the set level because:
      - no positional encoding is used,
      - self-attention is permutation-equivariant,
      - pooling is permutation-invariant.
    """

    def __init__(self, cfg: ZStateConfig, term_encoder: nn.Module) -> None:
        super().__init__(cfg, term_encoder)

        self.attention_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=cfg.d_term,
                    num_heads=cfg.attention_heads,
                    dropout=cfg.attention_dropout,
                )
                for _ in range(cfg.attention_layers)
            ]
        )

    def _encode_terms_to_spec(self, k_ids: torch.LongTensor, t_ids: torch.LongTensor, g_ids: torch.LongTensor, i_ids: torch.LongTensor, ) -> torch.Tensor:
        L = k_ids.size(0)
        if L == 0:
            return self.empty_state

        # [L, d_term]
        h_terms = self.term_encoder(k_ids, t_ids, g_ids, i_ids)

        # Add batch dimension: [1, L, d_term]
        x = h_terms.unsqueeze(0)

        # Interaction across terms
        for block in self.attention_layers:
            x = block(x)

        # Remove batch dimension: [L, d_term]
        h_terms_interacted = x.squeeze(0)

        # Pool interacted terms
        z_spec = self._pool_terms(h_terms_interacted)
        return z_spec


