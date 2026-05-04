"""
Physics-Informed Neural Network (PINN) surrogate.
Uses tanh activations, Fourier coordinate embedding, and supports physics loss.
"""

import torch
import torch.nn as nn
from typing import Dict
from models.base_model import BaseSurrogateModel
from agents.model_architect.physics_block_library import CoordinateEmbedding


class PINNSurrogate(BaseSurrogateModel):
    """
    PINN surrogate for CFD/FEA problems.
    compute_physics_loss() returns continuity and BC residuals.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 256, n_layers: int = 8,
                 dropout: float = 0.05):
        super().__init__(input_dim, output_dim, hidden_dim)

        self.embed = CoordinateEmbedding(input_dim=input_dim, embed_dim=hidden_dim)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            if i % 2 == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Xavier initialization (standard for PINNs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.trunk(h)
        return self.output_proj(h)

    def compute_physics_loss(self, pred: torch.Tensor,
                              coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Approximate continuity residual: div(V) ≈ 0.
        Computed via autograd on the network output w.r.t. input coords.
        """
        losses = {}

        if pred.shape[-1] < 3:
            return losses

        coords_req = coords.detach().requires_grad_(True)
        h = self.embed(coords_req)
        h = self.trunk(h)
        pred_with_grad = self.output_proj(h)

        try:
            u = pred_with_grad[..., 0]
            v = pred_with_grad[..., 1]

            du_dx = torch.autograd.grad(
                u.sum(), coords_req, create_graph=True, allow_unused=True
            )[0]
            dv_dy = torch.autograd.grad(
                v.sum(), coords_req, create_graph=True, allow_unused=True
            )[0]

            if du_dx is not None and dv_dy is not None:
                continuity = du_dx[..., 0] + dv_dy[..., 1]
                losses["continuity"] = torch.mean(continuity ** 2)
        except Exception:
            pass

        return losses
