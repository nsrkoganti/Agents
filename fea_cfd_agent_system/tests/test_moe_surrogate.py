"""
Tests for MixtureOfExpertsSurrogate.
Run: python -m pytest fea_cfd_agent_system/tests/test_moe_surrogate.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fea_cfd_agent_system.models.architectures.moe_surrogate import (
    MixtureOfExpertsSurrogate,
    NodeGatingNetwork,
)


# ── Minimal dummy expert ──────────────────────────────────────────────────────

class DummyExpert(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 6):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class PhysicsExpert(DummyExpert):
    def compute_physics_loss(self, pred=None, coords=None):
        return {"equilibrium": torch.tensor(0.05)}


def make_moe(n_experts=3, in_dim=3, out_dim=6, load_balance_weight=0.01):
    experts = [DummyExpert(in_dim, out_dim) for _ in range(n_experts)]
    return MixtureOfExpertsSurrogate(experts, in_dim=in_dim, out_dim=out_dim,
                                     load_balance_weight=load_balance_weight)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_shape():
    moe = make_moe()
    x = torch.randn(2, 64, 3)
    out = moe(x)
    assert out.shape == (2, 64, 6), f"Expected (2,64,6), got {out.shape}"


def test_gates_sum_to_one():
    moe = make_moe(n_experts=4)
    x = torch.randn(2, 50, 3)
    gates = moe.get_expert_attribution(x)
    assert gates.shape == (2, 50, 4)
    sums = gates.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Gates don't sum to 1: max_err={( sums - 1).abs().max():.6f}"


def test_no_nan_in_output():
    moe = make_moe()
    x = torch.randn(4, 128, 3)
    out = moe(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


def test_load_balance_loss_positive():
    experts = [PhysicsExpert(3, 6) for _ in range(3)]
    moe = MixtureOfExpertsSurrogate(experts, in_dim=3, out_dim=6)
    x = torch.randn(2, 32, 3)
    losses = moe.compute_physics_loss(x=x)
    assert "load_balance" in losses
    assert losses["load_balance"].item() >= 0.0, "Load balance loss must be non-negative"


def test_physics_loss_aggregation():
    experts = [PhysicsExpert(3, 6) for _ in range(2)]
    moe = MixtureOfExpertsSurrogate(experts, in_dim=3, out_dim=6)
    x = torch.randn(2, 32, 3)
    losses = moe.compute_physics_loss(x=x)
    # Each expert contributes its equilibrium loss weighted by gate prob
    physics_keys = [k for k in losses if "equilibrium" in k]
    assert len(physics_keys) == 2, f"Expected 2 expert physics losses, got {len(physics_keys)}"


def test_expert_attribution_shape():
    moe = make_moe(n_experts=3)
    x = torch.randn(1, 100, 3)
    attr = moe.get_expert_attribution(x)
    assert attr.shape == (1, 100, 3)


def test_freeze_unfreeze_experts():
    moe = make_moe()
    moe.freeze_experts()
    for expert in moe.experts:
        for p in expert.parameters():
            assert not p.requires_grad, "Expert should be frozen"

    moe.unfreeze_experts()
    for expert in moe.experts:
        for p in expert.parameters():
            assert p.requires_grad, "Expert should be unfrozen"

    # Gate should always be trainable
    for p in moe.gate.parameters():
        assert p.requires_grad


def test_gate_collapse_detection():
    """Verify that load balance loss penalises one-expert dominance."""
    moe = make_moe(n_experts=3, load_balance_weight=1.0)
    # Manually bias gate toward one expert
    with torch.no_grad():
        moe.gate.net[-1].bias.fill_(0)
        moe.gate.net[-1].bias[0] = 10.0  # dominant expert 0

    x = torch.randn(2, 32, 3)
    losses = moe.compute_physics_loss(x=x)
    lb = losses["load_balance"].item()
    assert lb > 0.01, f"Load balance should penalise collapse, got {lb:.6f}"


def test_requires_at_least_2_experts():
    with pytest.raises(AssertionError):
        MixtureOfExpertsSurrogate([DummyExpert()], in_dim=3, out_dim=6)


def test_node_gating_network_shape():
    gate = NodeGatingNetwork(in_dim=3, n_experts=5, hidden_dim=32)
    x = torch.randn(2, 64, 3)
    out = gate(x)
    assert out.shape == (2, 64, 5)
    assert torch.allclose(out.sum(-1), torch.ones(2, 64), atol=1e-5)
