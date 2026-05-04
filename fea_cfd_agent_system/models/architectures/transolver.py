"""
Transolver surrogate — physics attention over learned state slices.
Based on: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General Geometries", ICML 2024.
"""

import torch
import torch.nn as nn
from models.base_model import BaseSurrogateModel
from agents.model_architect.physics_block_library import (
    PhysicsAttentionBlock, CoordinateEmbedding
)


class TransolverSurrogate(BaseSurrogateModel):
    """
    Physics-attention transformer for unstructured mesh data.
    - Input: (B, N, input_dim) coordinates + features
    - Core: stack of PhysicsAttentionBlocks (O(S²) not O(N²))
    - Output: (B, N, output_dim) predicted fields
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 8,
                 n_slices: int = 32, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim, hidden_dim)

        self.embed = CoordinateEmbedding(
            input_dim=input_dim, embed_dim=hidden_dim
        )

        self.blocks = nn.ModuleList([
            PhysicsAttentionBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_slices=n_slices,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
