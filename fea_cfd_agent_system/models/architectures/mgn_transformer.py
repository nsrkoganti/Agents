"""
MeshGraphNet-Transformer — hybrid local MPNN + global Transformer for FEA.
arXiv:2601.23177, 2026

Architecture: Pre-MPNN (local) → Transformer (global) → Post-MPNN (local refinement)

Overcomes standard MGN limitation: inefficient long-range propagation.
Handles industrial-scale meshes for impact dynamics, nonlinear FEA,
and plasticity where deep message-passing stacks fail.

Benchmark: 2.6% relative L2 error on parametric I-beam FEA (best published result).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNBlock(nn.Module):
    """Local message passing block for mesh-based interaction."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst  = edge_index[0], edge_index[1]
        messages  = self.edge_mlp(
            torch.cat([x[src], x[dst], edge_attr], dim=-1)
        )
        # Mean aggregation
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        count = torch.zeros(x.shape[0], 1, device=x.device)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(len(dst), 1, device=x.device))
        agg = agg / (count + 1e-8)

        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(out + x)


class GlobalTransformerBlock(nn.Module):
    """
    Global Transformer processor — captures long-range interactions.
    Uses physics-attention (S slices) for O(N·S) instead of O(N²).
    """

    def __init__(self, hidden_dim: int, n_slices: int = 32, n_heads: int = 8):
        super().__init__()
        self.n_slices    = n_slices
        self.slice_query = nn.Linear(hidden_dim, n_slices)
        self.attn        = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1       = nn.LayerNorm(hidden_dim)
        self.norm2       = nn.LayerNorm(hidden_dim)
        self.ffn         = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) for single sample or (B, N, D) for batch
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        W = F.softmax(self.slice_query(x), dim=-1)          # (B, N, S)
        slice_tokens = torch.bmm(W.transpose(1, 2), x)      # (B, S, D)

        attn_out, _ = self.attn(slice_tokens, slice_tokens, slice_tokens)
        slice_tokens = self.norm1(slice_tokens + attn_out)
        slice_tokens = self.norm2(slice_tokens + self.ffn(slice_tokens))

        out = torch.bmm(W, slice_tokens) + x                # (B, N, D)
        return out.squeeze(0) if squeeze else out


class MeshGraphNetTransformer(nn.Module):
    """
    MeshGraphNet-Transformer for nonlinear / plastic FEA field prediction.

    Pipeline:
      1. Node + edge encoding
      2. n_pre_mpnn × MPNN blocks (local)
      3. 1 × Global Transformer block (long-range)
      4. n_post_mpnn × MPNN blocks (local refinement)
      5. Node decoding → field output

    Args:
        in_dim:       node input feature dimension
        out_dim:      output field channels
        hidden_dim:   model width
        n_pre_mpnn:   local MPNN blocks before Transformer
        n_post_mpnn:  local MPNN blocks after Transformer
        n_slices:     physics-attention slices in global Transformer
        n_heads:      attention heads
        edge_dim:     edge feature dimension
    """

    def __init__(self,
                 in_dim:      int = 3,
                 out_dim:     int = 9,
                 hidden_dim:  int = 128,
                 n_pre_mpnn:  int = 3,
                 n_post_mpnn: int = 3,
                 n_slices:    int = 32,
                 n_heads:     int = 8,
                 edge_dim:    int = 4):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pre_mpnn_blocks = nn.ModuleList([
            MPNNBlock(hidden_dim, hidden_dim) for _ in range(n_pre_mpnn)
        ])
        self.global_transformer = GlobalTransformerBlock(hidden_dim, n_slices, n_heads)
        self.post_mpnn_blocks = nn.ModuleList([
            MPNNBlock(hidden_dim, hidden_dim) for _ in range(n_post_mpnn)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        x:          (N, in_dim) node features
        coords:     (N, 3)
        edge_index: (2, E) mesh connectivity

        Returns: (N, out_dim)
        """
        if edge_index is None:
            edge_index = self._build_radius_graph(coords)

        edge_attr = self._compute_edge_features(coords, edge_index)
        edge_emb  = self.edge_encoder(edge_attr)
        h = self.node_encoder(x)

        for block in self.pre_mpnn_blocks:
            h = block(h, edge_index, edge_emb)

        h = self.global_transformer(h)  # (N, D)

        for block in self.post_mpnn_blocks:
            h = block(h, edge_index, edge_emb)

        return self.decoder(h)

    def _compute_edge_features(self, coords, edge_index):
        src, dst = edge_index
        rel  = coords[dst] - coords[src]
        dist = rel.norm(dim=-1, keepdim=True)
        return torch.cat([rel, dist], dim=-1)

    def _build_radius_graph(self, coords):
        from torch import cdist
        dist   = cdist(coords, coords)
        r      = dist.mean() * 0.5
        mask   = dist < r
        mask.fill_diagonal_(False)
        return mask.nonzero(as_tuple=False).T.contiguous()
