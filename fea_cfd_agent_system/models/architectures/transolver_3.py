"""
Transolver-3 — scales Transolver to industrial-scale FEA geometries.
arXiv:2602.04940 (2025)

Key innovations over Transolver++:
  - Eidetic state persistence across attention layers
  - Linear-scaling parallel GPU framework for 1M+ node meshes
  - Physics-attention with S=64 slices: O(N·S) instead of O(N²)
  - Designed for automotive, aircraft, and turbine FEA at production scale

Reference: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"
arXiv:2602.04940, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class EideticState(nn.Module):
    """Persistent physics memory that accumulates across attention layers."""

    def __init__(self, n_slices: int, hidden_dim: int):
        super().__init__()
        self.state = nn.Parameter(torch.zeros(1, n_slices, hidden_dim))
        self.gate  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        # slice_tokens: (B, S, D)
        state = self.state.expand(slice_tokens.size(0), -1, -1)
        gate  = torch.sigmoid(self.gate(slice_tokens))
        return slice_tokens + gate * state


class PhysicsAttentionBlock(nn.Module):
    """
    O(N·S + S²) attention — assigns mesh points to S physics-state slices,
    applies self-attention at slice level, then decodes back to mesh.
    """

    def __init__(self, hidden_dim: int, n_slices: int, n_heads: int,
                 use_eidetic: bool = True):
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
        self.eidetic = EideticState(n_slices, hidden_dim) if use_eidetic else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        W = F.softmax(self.slice_query(x), dim=-1)      # (B, N, S) soft assignment
        # Aggregate to slice tokens
        slice_tokens = torch.bmm(W.transpose(1, 2), x)  # (B, S, D)

        if self.eidetic is not None:
            slice_tokens = self.eidetic(slice_tokens)

        # Self-attention at slice level: O(S²)
        attn_out, _ = self.attn(slice_tokens, slice_tokens, slice_tokens)
        slice_tokens = self.norm1(slice_tokens + attn_out)
        slice_tokens = self.norm2(slice_tokens + self.ffn(slice_tokens))

        # Decode back to mesh: O(N·S)
        out = torch.bmm(W, slice_tokens)                 # (B, N, D)
        return out + x  # residual


class Transolver3(nn.Module):
    """
    Transolver-3 — industrial-scale FEA transformer surrogate.

    Args:
        in_dim:     input feature dimension (coordinates + BC features)
        out_dim:    output field dimension (e.g. 3 disp + 6 stress = 9)
        hidden_dim: internal representation width
        n_layers:   number of physics-attention blocks
        n_slices:   number of physics-state slices (S)
        n_heads:    attention heads (must divide hidden_dim)
    """

    def __init__(self,
                 in_dim:     int = 3,
                 out_dim:    int = 9,
                 hidden_dim: int = 256,
                 n_layers:   int = 8,
                 n_slices:   int = 64,
                 n_heads:    int = 8):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            PhysicsAttentionBlock(hidden_dim, n_slices, n_heads, use_eidetic=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim) — node coordinates + features
        Returns: (B, N, out_dim) — predicted field values
        """
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
