"""
Tests for novel architecture generation:
- ArchitectureDNA.from_llm_json()
- New block types (MambaBlock, ConvNeXtBlock, CrossAttentionBlock)
- NAS block-type search space expansion
- code_generator NaN + gradient validation
Run: python -m pytest fea_cfd_agent_system/tests/test_novel_architect.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fea_cfd_agent_system.agents.model_architect.architecture_dna import (
    ArchitectureDNA, ArchitectureBlock, BlockType
)
from fea_cfd_agent_system.agents.model_architect.physics_block_library import (
    MambaBlock, ConvNeXtBlock, CrossAttentionBlock,
    PhysicsAttentionBlock, CoordinateEmbedding
)


# ── BlockType enum ─────────────────────────────────────────────────────────────

def test_new_block_types_exist():
    assert BlockType.MAMBA_BLOCK     == "mamba_block"
    assert BlockType.CONV_NEXT_BLOCK == "conv_next_block"
    assert BlockType.CROSS_ATTENTION == "cross_attention"
    assert BlockType.SPECTRAL_NORM   == "spectral_norm"


# ── from_llm_json ──────────────────────────────────────────────────────────────

def make_llm_json(**overrides):
    base = {
        "name": "TestNovelNet",
        "family": "hybrid",
        "designed_for": "FEA_static_linear",
        "mesh_type": "any",
        "has_physics_loss": True,
        "physics_loss_types": ["equilibrium", "bc"],
        "parent_names": ["Transolver"],
        "input_blocks": [
            {"type": "coord_embed", "hidden_dim": 64},
        ],
        "core_blocks": [
            {"type": "mamba_block",      "hidden_dim": 64, "dropout": 0.1},
            {"type": "physics_attention","hidden_dim": 64, "n_slices": 8, "n_heads": 4},
            {"type": "cross_attention",  "hidden_dim": 64, "n_heads": 4, "n_queries": 16},
        ],
        "output_blocks": [
            {"type": "linear", "hidden_dim": 64},
            {"type": "linear", "hidden_dim": 4},
        ],
    }
    base.update(overrides)
    return base


def test_from_llm_json_basic():
    dna = ArchitectureDNA.from_llm_json(make_llm_json())
    assert dna.name == "TestNovelNet"
    assert dna.family == "hybrid"
    assert dna.has_physics_loss is True
    assert len(dna.input_processing) == 1
    assert len(dna.core_blocks) == 3
    assert len(dna.output_processing) == 2


def test_from_llm_json_block_types_parsed():
    dna = ArchitectureDNA.from_llm_json(make_llm_json())
    assert dna.core_blocks[0].block_type == BlockType.MAMBA_BLOCK
    assert dna.core_blocks[1].block_type == BlockType.PHYSICS_ATTN
    assert dna.core_blocks[2].block_type == BlockType.CROSS_ATTENTION


def test_from_llm_json_unknown_type_falls_back_to_linear():
    j = make_llm_json()
    j["core_blocks"][0]["type"] = "totally_unknown_block"
    dna = ArchitectureDNA.from_llm_json(j)
    assert dna.core_blocks[0].block_type == BlockType.LINEAR


def test_from_llm_json_roundtrip_serialisable():
    dna = ArchitectureDNA.from_llm_json(make_llm_json())
    d = dna.to_dict()
    dna2 = ArchitectureDNA.from_dict(d)
    assert dna2.name == dna.name
    assert len(dna2.core_blocks) == len(dna.core_blocks)


def test_from_llm_json_defaults_for_missing_keys():
    dna = ArchitectureDNA.from_llm_json({"name": "Minimal"})
    assert dna.family == "hybrid"
    assert dna.mesh_type == "any"
    assert dna.has_physics_loss is True
    assert dna.core_blocks == []


# ── New block forward passes ───────────────────────────────────────────────────

@pytest.fixture
def dummy_seq():
    return torch.randn(2, 64, 256)   # (B, N, D)


def test_mamba_block_shape(dummy_seq):
    block = MambaBlock(hidden_dim=256)
    out = block(dummy_seq)
    assert out.shape == dummy_seq.shape


def test_mamba_block_no_nan(dummy_seq):
    block = MambaBlock(hidden_dim=256)
    out = block(dummy_seq)
    assert torch.isfinite(out).all(), "MambaBlock output contains NaN/Inf"


def test_conv_next_block_shape(dummy_seq):
    block = ConvNeXtBlock(hidden_dim=256)
    out = block(dummy_seq)
    assert out.shape == dummy_seq.shape


def test_conv_next_block_no_nan(dummy_seq):
    block = ConvNeXtBlock(hidden_dim=256)
    out = block(dummy_seq)
    assert torch.isfinite(out).all()


def test_cross_attention_block_shape(dummy_seq):
    block = CrossAttentionBlock(hidden_dim=256, n_heads=8, n_queries=32)
    out = block(dummy_seq)
    assert out.shape == dummy_seq.shape


def test_cross_attention_block_no_nan(dummy_seq):
    block = CrossAttentionBlock(hidden_dim=256, n_heads=8, n_queries=32)
    out = block(dummy_seq)
    assert torch.isfinite(out).all()


def test_cross_attention_grad_flows(dummy_seq):
    block = CrossAttentionBlock(hidden_dim=256, n_heads=8, n_queries=16)
    out = block(dummy_seq)
    loss = out.mean()
    loss.backward()
    has_grad = any(
        p.grad is not None for p in block.parameters() if p.requires_grad
    )
    assert has_grad, "CrossAttentionBlock: no gradient reached any parameter"


# ── NAS engine — block type search ────────────────────────────────────────────

def test_nas_searchable_blocks_are_valid():
    from fea_cfd_agent_system.agents.model_architect.nas_engine import NASEngine
    valid_values = {bt.value for bt in BlockType}
    for bt in NASEngine.SEARCHABLE_CORE_BLOCKS:
        assert bt.value in valid_values, f"Unknown block type in NAS search space: {bt}"


def test_nas_novel_dna_block_type_override():
    """NAS _apply_params with block_types list must override core block types."""
    from fea_cfd_agent_system.agents.model_architect.nas_engine import NASEngine
    from fea_cfd_agent_system.agents.model_architect.architecture_dna import transolver_dna
    import copy

    engine = NASEngine({})
    dna = transolver_dna(hidden_dim=64, n_layers=4)
    dna.generation = 2  # mark as novel

    new_dna = engine._apply_params(
        dna,
        hidden_dim=64, n_layers=4, dropout=0.1,
        n_slices=8, n_heads=4,
        block_types=[BlockType.MAMBA_BLOCK, BlockType.CONV_NEXT_BLOCK],
    )
    # Block types should alternate: MAMBA, CONV_NEXT, MAMBA, CONV_NEXT
    assert new_dna.core_blocks[0].block_type == BlockType.MAMBA_BLOCK
    assert new_dna.core_blocks[1].block_type == BlockType.CONV_NEXT_BLOCK


# ── code_generator validation additions ───────────────────────────────────────

def test_code_generator_rejects_nan_output():
    """Generated model that produces NaN should fail validation."""
    from fea_cfd_agent_system.agents.model_architect.code_generator import CodeGenerator

    nan_model_code = """
import torch
import torch.nn as nn

class NaNModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.full((x.shape[0], x.shape[1], self.fc.out_features),
                          float('nan'))
"""
    gen = CodeGenerator.__new__(CodeGenerator)
    gen.llm = None  # Not needed for validate()
    is_valid, error = gen.validate(nan_model_code, input_dim=3, output_dim=4)
    assert not is_valid
    assert "NaN" in error or "Inf" in error


def test_code_generator_accepts_clean_model():
    """Well-behaved model should pass all validation checks."""
    from fea_cfd_agent_system.agents.model_architect.code_generator import CodeGenerator

    clean_code = """
import torch
import torch.nn as nn

class CleanModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.fc(x)
"""
    gen = CodeGenerator.__new__(CodeGenerator)
    gen.llm = None
    is_valid, error = gen.validate(clean_code, input_dim=3, output_dim=4)
    assert is_valid, f"Clean model failed validation: {error}"
