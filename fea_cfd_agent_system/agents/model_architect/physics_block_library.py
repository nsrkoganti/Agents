"""
Reusable physics-aware PyTorch layers.
These are the actual nn.Module implementations of the DNA building blocks.
The architect picks these and combines them into new models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordinateEmbedding(nn.Module):
    """
    Encodes (x, y, z) coordinates + additional features
    using Fourier feature mapping for better high-frequency learning.
    Used in: PINN, Transolver, any mesh-based model.
    """
    def __init__(self, input_dim: int = 3, embed_dim: int = 128,
                 n_fourier_features: int = 64):
        super().__init__()
        self.n_fourier = n_fourier_features
        self.register_buffer(
            "B",
            torch.randn(input_dim, n_fourier_features) * 10.0
        )
        self.proj = nn.Linear(2 * n_fourier_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xB = x @ self.B
        fourier_features = torch.cat([torch.sin(xB), torch.cos(xB)], dim=-1)
        return self.norm(self.proj(fourier_features))


class BoundaryConditionEncoder(nn.Module):
    """
    Encodes boundary condition information as additional input features.
    Tells the model WHICH boundary type every node belongs to.
    Inlet, outlet, wall, symmetry — each gets a learned embedding.
    """
    BC_TYPES = {"inlet": 0, "outlet": 1, "wall": 2, "symmetry": 3,
                "interior": 4, "unknown": 5}

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.bc_embedding   = nn.Embedding(len(self.BC_TYPES), hidden_dim // 4)
        self.bc_value_proj  = nn.Linear(4, hidden_dim // 4)
        self.proj           = nn.Linear(hidden_dim // 2, hidden_dim)
        self.norm           = nn.LayerNorm(hidden_dim)

    def forward(self, bc_type_ids: torch.Tensor,
                bc_values: torch.Tensor) -> torch.Tensor:
        type_emb  = self.bc_embedding(bc_type_ids)
        value_emb = self.bc_value_proj(bc_values)
        combined  = torch.cat([type_emb, value_emb], dim=-1)
        return self.norm(self.proj(combined))


class PhysicsAttentionBlock(nn.Module):
    """
    TRANSOLVER-STYLE PHYSICS ATTENTION (Wu et al. ICML 2024).

    Instead of attending over N mesh points (O(N^2) cost),
    learn S "physical state" slices and attend over slices (O(S^2), S << N).

    Architecture:
    1. Slice-weight matrix W: maps N mesh points -> S slices (soft assignment)
    2. Encode each slice into a physics-aware token
    3. Multi-head attention ACROSS slices
    4. Decode slice tokens back to mesh points via W^T
    """

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8,
                 n_slices: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.n_slices   = n_slices
        self.head_dim   = hidden_dim // n_heads

        self.slice_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_slices),
            nn.Softmax(dim=-1),
        )

        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, hidden_dim) — features at each mesh point
        Returns: (B, N, hidden_dim) — updated features at each mesh point
        """
        B, N, D  = x.shape
        residual = x

        # 1. Compute slice assignment weights
        W = self.slice_weight(x)           # (B, N, S)

        # 2. Aggregate mesh points into physics-aware slice tokens
        slice_tokens = torch.bmm(W.transpose(1, 2), x)  # (B, S, D)

        # 3. Multi-head attention ACROSS slices
        attn_out, _ = self.attn(slice_tokens, slice_tokens, slice_tokens)
        slice_tokens = self.norm1(slice_tokens + self.dropout(attn_out))

        # 4. FFN on slice tokens
        ffn_out      = self.ffn(slice_tokens)
        slice_tokens = self.norm2(slice_tokens + self.dropout(ffn_out))

        # 5. Decode slice tokens back to mesh points
        x_out = torch.bmm(W, slice_tokens)  # (B, N, D)

        return x_out + residual


class FourierLayer(nn.Module):
    """
    FNO-STYLE FOURIER LAYER.
    Learns in spectral space. Very efficient for structured grids.

    Architecture:
    1. FFT the input -> spectral coefficients
    2. Multiply by learned complex weights (only low modes)
    3. IFFT back to physical space
    4. Add local linear transform (bypasses spectral path)
    """

    def __init__(self, hidden_dim: int = 64, n_modes: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes    = n_modes

        self.weights_real = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes, n_modes) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes, n_modes) * 0.01
        )
        self.w = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) — structured grid input"""
        B, C, H, W = x.shape

        x_ft   = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        m1 = min(self.n_modes, H // 2 + 1)
        m2 = min(self.n_modes, W // 2 + 1)

        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            weights[:, :, :m1, :m2]
        )
        x_spectral = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        x_local    = self.w(x)

        return F.gelu(x_spectral + x_local)


class GraphConvBlock(nn.Module):
    """
    GNN MESSAGE PASSING BLOCK — MeshGraphNet style.
    Nodes receive messages from neighbors along mesh edges.
    """

    def __init__(self, hidden_dim: int = 128, edge_dim: int = 4):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        node_features: (N, hidden_dim)
        edge_index:    (2, E)
        edge_features: (E, edge_dim)
        Returns: (N, hidden_dim)
        """
        src, dst = edge_index[0], edge_index[1]

        messages = self.edge_mlp(torch.cat([
            node_features[src],
            node_features[dst],
            edge_features,
            (node_features[src] - node_features[dst])
        ], dim=-1))

        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)
        count = torch.zeros(node_features.shape[0], 1, device=node_features.device)
        count.index_add_(0, dst, torch.ones(messages.shape[0], 1, device=messages.device))
        aggregated = aggregated / (count + 1e-8)

        updated = self.node_mlp(torch.cat([node_features, aggregated], dim=-1))
        return updated + node_features


# ── Novel blocks for LLM-designed architectures ───────────────────────────────

class MambaBlock(nn.Module):
    """
    Simplified selective state-space block inspired by Mamba (Gu & Dao, 2023).
    Achieves O(N) complexity vs O(N^2) for attention — ideal for large FEA meshes.
    Uses input-dependent gating rather than a full SSM scan for simplicity.
    Interface: (B, N, hidden_dim) -> (B, N, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, d_state: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_state    = d_state

        # Input projection (expand for gating)
        self.in_proj    = nn.Linear(hidden_dim, hidden_dim * 2)
        # SSM parameters (simplified: input-dependent B, C, delta)
        self.x_proj     = nn.Linear(hidden_dim, d_state * 2 + 1)
        self.dt_proj    = nn.Linear(1, hidden_dim)
        self.A_log      = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(hidden_dim, -1)))
        self.D          = nn.Parameter(torch.ones(hidden_dim))
        # Output
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, hidden_dim)"""
        residual = x
        xz = self.in_proj(x)                          # (B, N, 2D)
        x_inner, z = xz.chunk(2, dim=-1)              # each (B, N, D)

        # Input-dependent parameters
        x_dbl  = self.x_proj(x_inner)                 # (B, N, d_state*2+1)
        dt_raw = x_dbl[..., :1]                       # (B, N, 1)
        B_raw  = x_dbl[..., 1:self.d_state + 1]      # (B, N, d_state)
        C_raw  = x_dbl[..., self.d_state + 1:]       # (B, N, d_state)

        dt  = F.softplus(self.dt_proj(dt_raw))        # (B, N, D)
        A   = -torch.exp(self.A_log)                  # (D, d_state)

        # Discretized SSM (parallel, simplified — treats N as sequence)
        # Use cumulative product along sequence dim as proxy for scan
        dA  = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,N,D,d_state)
        dB  = dt.unsqueeze(-1) * B_raw.unsqueeze(2)                      # (B,N,D,d_state)

        # Simplified selective scan: exponential weighted cumsum
        h = (x_inner.unsqueeze(-1) * dB)             # (B, N, D, d_state)
        h = h.cumsum(dim=1)                           # aggregate over sequence
        y = (h * C_raw.unsqueeze(2)).sum(-1)          # (B, N, D)
        y = y + self.D * x_inner                     # skip connection

        # Gate with z (SiLU)
        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        return self.norm(out + residual)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block adapted for point-cloud / mesh sequences.
    Uses depthwise Conv1d (per-channel, local neighbourhood) + pointwise MLPs.
    Much more parameter-efficient than attention for local feature extraction.
    Interface: (B, N, hidden_dim) -> (B, N, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, kernel_size: int = 7,
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm       = nn.LayerNorm(hidden_dim)
        # Depthwise conv over the node dimension (treated as sequence)
        self.dw_conv    = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size,
            padding=kernel_size // 2, groups=hidden_dim
        )
        self.pw_expand  = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.pw_shrink  = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.act        = nn.GELU()
        self.dropout    = nn.Dropout(dropout)
        # Learnable layer scale (stabilises training)
        self.gamma      = nn.Parameter(torch.ones(hidden_dim) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, hidden_dim)"""
        residual = x
        x = self.norm(x)
        # Depthwise conv: needs (B, C, N)
        x = self.dw_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act(self.pw_expand(x))
        x = self.dropout(self.pw_shrink(x))
        return residual + self.gamma * x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between mesh nodes (keys/values) and physics query tokens.
    Extends the Transolver idea: instead of self-attention over slices,
    queries come from a separate learnable physics-state embedding.
    Allows the model to attend from any query point to the full mesh.
    Interface: (B, N, hidden_dim) -> (B, N, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8,
                 n_queries: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_queries   = n_queries
        # Learnable physics state queries
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        # Cross-attention: queries attend to mesh nodes
        self.cross_attn  = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        # Project physics states back to node-wise features
        self.decode_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, hidden_dim)"""
        B = x.shape[0]
        residual = x

        # Expand learnable queries to batch
        q = self.query_embed.expand(B, -1, -1)     # (B, n_queries, D)

        # Cross-attend: physics queries attend over mesh nodes
        q_out, _ = self.cross_attn(q, x, x)        # (B, n_queries, D)
        q_out = self.norm1(q + q_out)

        # FFN on physics states
        q_out = self.norm2(q_out + self.ffn(q_out))

        # Decode back to node resolution: mesh nodes attend over physics states
        x_out, _ = self.decode_attn(x, q_out, q_out)  # (B, N, D)
        x_out = self.norm3(x_out + residual)

        return x_out
