"""
Hybrid Transolver-PINN model.
Combines physics-attention (for global structure) with PINN-style physics loss.
Best choice when standard models fail on complex turbulent flows.
"""

import torch
import torch.nn as nn
from typing import Dict
from models.base_model import BaseSurrogateModel
from agents.model_architect.physics_block_library import (
    PhysicsAttentionBlock, CoordinateEmbedding
)


class HybridTransolverPINN(BaseSurrogateModel):
    """
    Hybrid architecture:
    1. Fourier coordinate embedding
    2. Physics-attention layers (Transolver style)
    3. PINN-style physics loss via autograd
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 8,
                 n_slices: int = 32, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(input_dim, output_dim, hidden_dim)

        self.embed = CoordinateEmbedding(
            input_dim=input_dim, embed_dim=hidden_dim
        )

        self.attn_layers = nn.ModuleList([
            PhysicsAttentionBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_slices=n_slices,
                dropout=dropout,
            )
            for _ in range(n_layers // 2)
        ])

        self.pinn_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            ) for _ in range(n_layers // 2)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.attn_layers:
            h = block(h)
        h = self.pinn_layers(h)
        return self.output_proj(h)

    def compute_physics_loss(self, pred: torch.Tensor,
                              coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Continuity residual loss via autograd."""
        losses = {}
        if pred.shape[-1] < 2:
            return losses

        try:
            coords_req = coords.detach().requires_grad_(True)
            h = self.embed(coords_req)
            for block in self.attn_layers:
                h = block(h)
            h = self.pinn_layers(h)
            pred_wg = self.output_proj(h)

            u = pred_wg[..., 0].sum()
            v = pred_wg[..., 1].sum()

            du = torch.autograd.grad(u, coords_req, create_graph=True,
                                     allow_unused=True)[0]
            dv = torch.autograd.grad(v, coords_req, create_graph=True,
                                     allow_unused=True)[0]

            if du is not None and dv is not None:
                div = du[..., 0] + dv[..., 1]
                losses["continuity"] = torch.mean(div ** 2)
        except Exception:
            pass

        return losses
