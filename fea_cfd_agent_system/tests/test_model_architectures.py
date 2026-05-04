"""Tests for all surrogate model architectures."""

import pytest
import torch


BATCH = 2
N     = 100   # mesh nodes
DIM   = 3     # input dimension (x, y, z)
OUT   = 4     # output fields (u, v, w, p)


def _dummy_input():
    return torch.randn(BATCH, N, DIM)


def test_mlp_surrogate():
    from models.architectures.mlp_surrogate import MLPSurrogate
    model = MLPSurrogate(input_dim=DIM, output_dim=OUT, hidden_dim=64, n_layers=3)
    out = model(_dummy_input())
    assert out.shape == (BATCH, N, OUT)
    assert not torch.any(torch.isnan(out))


def test_transolver():
    from models.architectures.transolver import TransolverSurrogate
    model = TransolverSurrogate(
        input_dim=DIM, output_dim=OUT,
        hidden_dim=64, n_layers=2, n_slices=8, n_heads=4
    )
    out = model(_dummy_input())
    assert out.shape == (BATCH, N, OUT)
    assert not torch.any(torch.isnan(out))


def test_pinn():
    from models.architectures.pinn import PINNSurrogate
    model = PINNSurrogate(input_dim=DIM, output_dim=OUT, hidden_dim=64, n_layers=4)
    out = model(_dummy_input())
    assert out.shape == (BATCH, N, OUT)
    assert not torch.any(torch.isnan(out))


def test_hybrid():
    from models.architectures.hybrid_model import HybridTransolverPINN
    model = HybridTransolverPINN(
        input_dim=DIM, output_dim=OUT,
        hidden_dim=64, n_layers=4, n_slices=8, n_heads=4
    )
    out = model(_dummy_input())
    assert out.shape == (BATCH, N, OUT)
    assert not torch.any(torch.isnan(out))


def test_gnn_fallback():
    from models.architectures.gnn_surrogate import GNNSurrogate
    model = GNNSurrogate(input_dim=DIM, output_dim=OUT, hidden_dim=64, n_layers=2)
    # No graph provided — falls back to MLP
    out = model(_dummy_input(), edge_index=None, edge_features=None)
    assert out.shape == (BATCH, N, OUT)


def test_pinn_physics_loss():
    from models.architectures.pinn import PINNSurrogate
    model = PINNSurrogate(input_dim=3, output_dim=4, hidden_dim=32, n_layers=2)
    x = torch.randn(1, 20, 3)
    pred = model(x)
    losses = model.compute_physics_loss(pred, x)
    if "continuity" in losses:
        assert losses["continuity"].item() >= 0


def test_model_n_params():
    from models.architectures.transolver import TransolverSurrogate
    model = TransolverSurrogate(hidden_dim=128, n_layers=4)
    n = model.n_parameters()
    assert n > 0
    assert n < 100_000_000
