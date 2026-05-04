"""
FNO-style surrogate for structured grid data.
Operates in spectral space — very efficient for regular meshes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseSurrogateModel


class FNOSurrogate(BaseSurrogateModel):
    """
    Fourier Neural Operator for structured (grid) CFD/FEA data.
    Input must be (B, C, H, W) — use MeshConverter.to_structured_grid() first.
    For unstructured data, TransolverSurrogate is preferred.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 64, n_layers: int = 4,
                 n_modes: int = 16):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.n_modes  = n_modes
        self.n_layers = n_layers

        self.lift = nn.Conv2d(input_dim, hidden_dim, 1)

        self.spectral_layers = nn.ModuleList([
            _SpectralConv2d(hidden_dim, hidden_dim, n_modes)
            for _ in range(n_layers)
        ])
        self.bypass_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1)
            for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) point cloud — reshape for FNO
        if x.ndim == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            if H * W != N:
                raise ValueError(
                    f"FNO requires structured grid: N={N} is not a perfect square. "
                    "Use TransolverSurrogate for unstructured meshes."
                )
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.lift(x)
        for spec, bypass in zip(self.spectral_layers, self.bypass_layers):
            x = F.gelu(spec(x) + bypass(x))
        return self.proj(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.output_dim)


class _SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.n_modes = n_modes

        self.wr = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, n_modes) * 0.02
        )
        self.wi = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, n_modes) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft   = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1 = min(self.n_modes, H // 2 + 1)
        m2 = min(self.n_modes, W // 2 + 1)
        w  = torch.complex(self.wr, self.wi)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            w[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
