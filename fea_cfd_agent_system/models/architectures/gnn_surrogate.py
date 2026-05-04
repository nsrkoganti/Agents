"""GNN surrogate using MeshGraphNet-style message passing."""

import torch
import torch.nn as nn
from typing import Optional
from models.base_model import BaseSurrogateModel
from agents.model_architect.physics_block_library import GraphConvBlock


class GNNSurrogate(BaseSurrogateModel):
    """
    Graph Neural Network surrogate for unstructured mesh CFD/FEA.
    Requires edge_index and edge_features as additional inputs.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 128, n_layers: int = 6,
                 edge_dim: int = 4, dropout: float = 0.1):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.edge_dim = edge_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.conv_layers = nn.ModuleList([
            GraphConvBlock(hidden_dim=hidden_dim, edge_dim=hidden_dim)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, input_dim) or (N, input_dim)
        For batch mode, processes each sample independently.
        """
        if x.ndim == 3:
            # Process per sample in batch
            B, N, D = x.shape
            outs = []
            for b in range(B):
                out_b = self._forward_single(
                    x[b], edge_index, edge_features
                )
                outs.append(out_b)
            return torch.stack(outs, dim=0)
        else:
            return self._forward_single(x, edge_index, edge_features)

    def _forward_single(self, x: torch.Tensor,
                         edge_index: Optional[torch.Tensor],
                         edge_feats: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.node_encoder(x)

        if edge_index is None or edge_feats is None:
            # No graph structure — fall back to MLP
            for _ in self.conv_layers:
                h = h + nn.functional.gelu(h)
        else:
            e = self.edge_encoder(edge_feats)
            for conv in self.conv_layers:
                h = conv(h, edge_index, e)

        return self.output_proj(h)
