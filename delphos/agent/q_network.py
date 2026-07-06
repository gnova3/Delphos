# =============================================================================
#   DQN head for learned state representation
#
#       Q-network mapping z(s) -> Q(s, :)
#
#   This head predicts action values over the shared global action catalogue.
# =============================================================================
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.init as init


class QNetwork(nn.Module):
    """Feed-forward Deep Q-Network head.

    The Q-network maps an encoded latent state representation z(s) into action 
    values Q(s,a) for every action a in the global Delphos action catalogue.
    
    Attributes:
        model (nn.Sequential): Multi-layer perceptron used to approximate the action-value function.

    Args:
        input_size: Dimension of the latent state representation z(s) [int].
        output_size: Number of actions in the global action catalogue [int].
        hidden_layers: Hidden layer sizes of the MLP [Sequence[int]].

    Raises:
        ValueError: If any network dimension is invalid.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: Sequence[int] = (128, 64),) -> None:        
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}.")
        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers must contain at least one layer.")

        layers: list[nn.Module] = []
        last_size = input_size

        for hidden_size in hidden_layers:
            if hidden_size <= 0:
                raise ValueError(f"Hidden layer sizes must be positive, got {hidden_size}.")
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size

        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action-value estimates.

        Evaluates the Q-network and returns predicted Q-values for all actions.

        Args:
            x: Input latent state representation [state_dim] or [batch_size, state_dim]
        
        Returns:
            torch.Tensor: Predicted Q-values [num_actions] or [batch_size, num_actions]

        Raises:
            ValueError: If the input shape is incompatible with the network.
        """
        if x.dim() == 1:
            if x.numel() != self.model[0].in_features:
                raise ValueError(f"Expected 1D input of size {self.model[0].in_features}, got {x.numel()}.")
        elif x.dim() == 2:
            if x.size(1) != self.model[0].in_features:
                raise ValueError(f"Expected input shape [batch, {self.model[0].in_features}], got {tuple(x.shape)}.")
        else:
            raise ValueError(f"Input must be 1D or 2D, got shape {tuple(x.shape)}.")
        return self.model(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize network parameters.

        Linear layers use Kaiming uniform initialization to improve training
        stability when combined with ReLU activations.
        Bias terms are initialized to zero.
        """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(
                module.weight,
                a=0.0,
                mode="fan_in",
                nonlinearity="relu",
            )
            if module.bias is not None:
                init.zeros_(module.bias)

    def summary(self) -> dict:
        """Return a compact network architecture summary.

        Returns:
            dict:
                - input_size
                - output_size
                - hidden_layers
                - n_parameters
            """
        linear_layers = [m for m in self.model if isinstance(m, nn.Linear)]
        return {
            "input_size": linear_layers[0].in_features,
            "output_size": linear_layers[-1].out_features,
            "hidden_layers": [m.out_features for m in linear_layers[:-1]],
            "n_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }