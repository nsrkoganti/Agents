"""
Smoke tests for FEA surrogate ML architectures.
Tests Transolver-3, Transolver++, EAGNN, MeshGraphNet-Transformer, GNSS, GS-PI-DeepONet,
and Factorized FNO (forward pass, output shape, parameter counts).
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cpu")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def point_cloud():
    N, C = 64, 3
    return torch.randn(N, C, device=DEVICE)


@pytest.fixture
def batched_point_cloud():
    B, N, C = 2, 64, 3
    return torch.randn(B, N, C, device=DEVICE)


@pytest.fixture
def edge_index():
    N = 64
    src = torch.randint(0, N, (320,))
    dst = torch.randint(0, N, (320,))
    return torch.stack([src, dst], dim=0).to(DEVICE)


# ── Transolver-3 ──────────────────────────────────────────────────────────────

class TestTransolver3:

    def test_import(self):
        from models.architectures.transolver_3 import Transolver3
        assert Transolver3 is not None

    def test_forward_batched(self, batched_point_cloud):
        from models.architectures.transolver_3 import Transolver3
        model = Transolver3(in_dim=3, out_dim=6, hidden_dim=32,
                            n_layers=2, n_slices=8, n_heads=2).to(DEVICE)
        out = model(batched_point_cloud)
        B, N, _ = batched_point_cloud.shape
        assert out.shape == (B, N, 6), f"Unexpected shape: {out.shape}"

    def test_no_nan_output(self, batched_point_cloud):
        from models.architectures.transolver_3 import Transolver3
        model = Transolver3(in_dim=3, out_dim=6, hidden_dim=32,
                            n_layers=2, n_slices=8, n_heads=2).to(DEVICE)
        out = model(batched_point_cloud)
        assert not torch.isnan(out).any(), "NaN in Transolver-3 output"

    def test_param_count(self):
        from models.architectures.transolver_3 import Transolver3
        model = Transolver3(in_dim=3, out_dim=6, hidden_dim=64,
                            n_layers=4, n_slices=16, n_heads=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000, f"Too few params: {n_params}"
        assert n_params < 50_000_000, f"Suspiciously many params: {n_params}"

    def test_eidetic_state_exists(self):
        from models.architectures.transolver_3 import Transolver3, EideticState
        assert EideticState is not None
        model = Transolver3(in_dim=3, out_dim=6)
        # EideticState parameters should be in the model
        param_names = [n for n, _ in model.named_parameters()]
        assert len(param_names) > 0


# ── Transolver++ ──────────────────────────────────────────────────────────────

class TestTransolverPP:

    def test_import(self):
        from models.architectures.transolver_pp import TransolverPP
        assert TransolverPP is not None

    def test_forward_batched(self, batched_point_cloud):
        from models.architectures.transolver_pp import TransolverPP
        model = TransolverPP(in_dim=3, out_dim=9, hidden_dim=32,
                             n_layers=2, n_slices=8, n_heads=2).to(DEVICE)
        out = model(batched_point_cloud)
        B, N, _ = batched_point_cloud.shape
        assert out.shape == (B, N, 9), f"Unexpected shape: {out.shape}"

    def test_no_nan_output(self, batched_point_cloud):
        from models.architectures.transolver_pp import TransolverPP
        model = TransolverPP(in_dim=3, out_dim=9, hidden_dim=32,
                             n_layers=2, n_slices=8, n_heads=2).to(DEVICE)
        out = model(batched_point_cloud)
        assert not torch.isnan(out).any(), "NaN in Transolver++ output"

    def test_larger_out_dim(self, batched_point_cloud):
        from models.architectures.transolver_pp import TransolverPP
        model = TransolverPP(in_dim=3, out_dim=12, hidden_dim=32,
                             n_layers=2, n_slices=8, n_heads=2).to(DEVICE)
        out = model(batched_point_cloud)
        assert out.shape[-1] == 12


# ── EAGNN ─────────────────────────────────────────────────────────────────────

class TestEAGNN:

    def test_import(self):
        from models.architectures.eagnn import EAGNN
        assert EAGNN is not None

    def test_forward_no_edge_index(self, point_cloud):
        from models.architectures.eagnn import EAGNN
        model = EAGNN(in_dim=3, out_dim=9, hidden_dim=32,
                      n_layers=2, p_aug=0.0).to(DEVICE)
        coords = point_cloud
        out = model(point_cloud, coords)
        assert out.shape == (len(point_cloud), 9), f"Unexpected shape: {out.shape}"

    def test_forward_with_edge_index(self, point_cloud, edge_index):
        from models.architectures.eagnn import EAGNN
        model = EAGNN(in_dim=3, out_dim=6, hidden_dim=32,
                      n_layers=2, p_aug=0.0).to(DEVICE)
        out = model(point_cloud, point_cloud, edge_index)
        assert out.shape == (len(point_cloud), 6)

    def test_augmentation_edges_added(self, point_cloud):
        from models.architectures.eagnn import EAGNN
        model = EAGNN(in_dim=3, out_dim=6, hidden_dim=16,
                      n_layers=1, p_aug=0.05, n_aug_edges=10).to(DEVICE)
        N = len(point_cloud)
        # Build graph manually to test augmentation
        base_idx = torch.stack([torch.arange(N-1), torch.arange(1, N)], dim=0)
        aug_idx = model.build_graph(point_cloud, base_idx)
        assert aug_idx.shape[1] > base_idx.shape[1]

    def test_no_nan_output(self, point_cloud):
        from models.architectures.eagnn import EAGNN
        model = EAGNN(in_dim=3, out_dim=6, hidden_dim=32,
                      n_layers=2, p_aug=0.01).to(DEVICE)
        out = model(point_cloud, point_cloud)
        assert not torch.isnan(out).any()


# ── MeshGraphNet-Transformer ──────────────────────────────────────────────────

class TestMeshGraphNetTransformer:

    def test_import(self):
        from models.architectures.mgn_transformer import MeshGraphNetTransformer
        assert MeshGraphNetTransformer is not None

    def test_forward_no_edge_index(self, point_cloud):
        from models.architectures.mgn_transformer import MeshGraphNetTransformer
        model = MeshGraphNetTransformer(in_dim=3, out_dim=9, hidden_dim=32,
                                        n_pre_mpnn=1, n_post_mpnn=1,
                                        n_slices=8, n_heads=2).to(DEVICE)
        out = model(point_cloud, point_cloud)
        assert out.shape == (len(point_cloud), 9)

    def test_forward_with_edge_index(self, point_cloud, edge_index):
        from models.architectures.mgn_transformer import MeshGraphNetTransformer
        model = MeshGraphNetTransformer(in_dim=3, out_dim=6, hidden_dim=32,
                                        n_pre_mpnn=1, n_post_mpnn=1,
                                        n_slices=8, n_heads=2).to(DEVICE)
        out = model(point_cloud, point_cloud, edge_index)
        assert out.shape == (len(point_cloud), 6)

    def test_no_nan_output(self, point_cloud):
        from models.architectures.mgn_transformer import MeshGraphNetTransformer
        model = MeshGraphNetTransformer(in_dim=3, out_dim=6, hidden_dim=32,
                                        n_pre_mpnn=1, n_post_mpnn=1,
                                        n_slices=8, n_heads=2).to(DEVICE)
        out = model(point_cloud, point_cloud)
        assert not torch.isnan(out).any()

    def test_global_transformer_shapes(self):
        from models.architectures.mgn_transformer import GlobalTransformerBlock
        block = GlobalTransformerBlock(hidden_dim=32, n_slices=8, n_heads=2)
        x = torch.randn(50, 32)
        out = block(x)
        assert out.shape == (50, 32)


# ── GNSS ──────────────────────────────────────────────────────────────────────

class TestGNSS:

    def test_import(self):
        from models.architectures.gnss import GNSS
        assert GNSS is not None

    def test_forward_auto_graph(self, point_cloud):
        from models.architectures.gnss import GNSS
        N = len(point_cloud)
        node_feats = torch.randn(N, 9)   # coords + velocity + acceleration
        coords = point_cloud
        model = GNSS(in_dim=9, out_dim=3, hidden_dim=32, n_layers=2).to(DEVICE)
        out = model(node_feats, coords)
        assert out.shape == (N, 3)

    def test_forward_with_edge_index(self, point_cloud, edge_index):
        from models.architectures.gnss import GNSS
        N = len(point_cloud)
        node_feats = torch.randn(N, 9)
        model = GNSS(in_dim=9, out_dim=3, hidden_dim=32, n_layers=2).to(DEVICE)
        out = model(node_feats, point_cloud, edge_index)
        assert out.shape == (N, 3)

    def test_no_nan_output(self, point_cloud):
        from models.architectures.gnss import GNSS
        N = len(point_cloud)
        node_feats = torch.randn(N, 9)
        model = GNSS(in_dim=9, out_dim=3, hidden_dim=32, n_layers=2).to(DEVICE)
        out = model(node_feats, point_cloud)
        assert not torch.isnan(out).any()

    def test_wavelength_graph_builds(self, point_cloud):
        from models.architectures.gnss import GNSS
        model = GNSS()
        edge_index = model._wavelength_graph(point_cloud, char_wavelength=0.5)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_edge_features_shape(self, point_cloud, edge_index):
        from models.architectures.gnss import GNSS
        model = GNSS()
        feats = model.compute_edge_features(point_cloud, edge_index)
        assert feats.shape[0] == edge_index.shape[1]
        assert feats.shape[1] == 5   # rel(3) + dist(1) + wavelength_ratio(1)


# ── GS-PI-DeepONet ────────────────────────────────────────────────────────────

class TestGSPIDeepONet:

    def test_import(self):
        from models.architectures.deeponet import GSPIDeepONet
        assert GSPIDeepONet is not None

    def test_forward_shape(self):
        from models.architectures.deeponet import GSPIDeepONet
        model = GSPIDeepONet(coord_dim=3, param_dim=4, out_dim=9,
                              hidden_dim=32, basis_dim=16).to(DEVICE)
        N = 50   # mesh nodes
        M = 20   # query points
        coords       = torch.randn(N, 3)
        query_points = torch.randn(M, 3)
        params       = torch.randn(4)
        out = model(coords, query_points, params)
        assert out.shape == (M, 9), f"Unexpected shape: {out.shape}"

    def test_batched_params(self):
        from models.architectures.deeponet import GSPIDeepONet
        model = GSPIDeepONet(coord_dim=3, param_dim=4, out_dim=6,
                              hidden_dim=32, basis_dim=16).to(DEVICE)
        coords       = torch.randn(30, 3)
        query_points = torch.randn(15, 3)
        params       = torch.randn(1, 4)  # batched
        out = model(coords, query_points, params)
        assert out.shape == (15, 6)

    def test_physics_loss_returns_scalar(self):
        from models.architectures.deeponet import GSPIDeepONet
        model = GSPIDeepONet()
        M = 10
        stress = torch.randn(M, 6)
        strain = torch.randn(M, 6)
        loss = model.physics_loss(None, stress, strain)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_no_nan_output(self):
        from models.architectures.deeponet import GSPIDeepONet
        model = GSPIDeepONet(coord_dim=3, param_dim=4, out_dim=9,
                              hidden_dim=32, basis_dim=16).to(DEVICE)
        coords       = torch.randn(30, 3)
        query_points = torch.randn(15, 3)
        params       = torch.randn(4)
        out = model(coords, query_points, params)
        assert not torch.isnan(out).any()


# ── Factorized FNO ────────────────────────────────────────────────────────────

class TestFactorizedFNO:

    def test_import(self):
        from models.architectures.fno_surrogate import FactorizedFNOSurrogate, FNOSurrogate
        assert FactorizedFNOSurrogate is not None
        assert FNOSurrogate is FactorizedFNOSurrogate  # backward compat alias

    def test_forward_structured_grid(self):
        from models.architectures.fno_surrogate import FactorizedFNOSurrogate
        model = FactorizedFNOSurrogate(input_dim=3, output_dim=6, hidden_dim=16,
                                        n_layers=2, n_modes_x=8, n_modes_y=8).to(DEVICE)
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 6, 16, 16), f"Unexpected shape: {out.shape}"

    def test_forward_point_cloud_perfect_square(self):
        from models.architectures.fno_surrogate import FactorizedFNOSurrogate
        model = FactorizedFNOSurrogate(input_dim=3, output_dim=6, hidden_dim=16,
                                        n_layers=2, n_modes_x=8, n_modes_y=8).to(DEVICE)
        # N=256 = 16×16 perfect square
        x = torch.randn(2, 256, 3)
        out = model(x)
        assert out.shape == (2, 256, 6), f"Unexpected shape: {out.shape}"

    def test_non_square_raises(self):
        from models.architectures.fno_surrogate import FactorizedFNOSurrogate
        model = FactorizedFNOSurrogate(input_dim=3, output_dim=6, hidden_dim=16,
                                        n_layers=2).to(DEVICE)
        x = torch.randn(2, 100, 3)  # 100 is not a perfect square
        with pytest.raises(ValueError):
            model(x)

    def test_no_nan_output(self):
        from models.architectures.fno_surrogate import FactorizedFNOSurrogate
        model = FactorizedFNOSurrogate(input_dim=3, output_dim=6, hidden_dim=16,
                                        n_layers=2).to(DEVICE)
        x = torch.randn(1, 3, 8, 8)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_factorized_spectral_conv(self):
        from models.architectures.fno_surrogate import FactorizedSpectralConv2d
        conv = FactorizedSpectralConv2d(in_ch=4, out_ch=4, n_modes_x=4, n_modes_y=4)
        x = torch.randn(1, 4, 8, 8)
        out = conv(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
