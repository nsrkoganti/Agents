"""Simple MLP surrogate — fallback when nothing else works."""

import torch
import torch.nn as nn
from models.base_model import BaseSurrogateModel


class MLPSurrogate(BaseSurrogateModel):
    """
    Fully-connected MLP surrogate.
    Not great for complex physics, but always works as a baseline.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim, hidden_dim)

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
