"""Tests for physics building blocks."""

import pytest
import torch
import numpy as np


def test_coordinate_embedding():
    from agents.model_architect.physics_block_library import CoordinateEmbedding
    emb = CoordinateEmbedding(input_dim=3, embed_dim=128)
    x = torch.randn(2, 50, 3)
    out = emb(x)
    assert out.shape == (2, 50, 128), f"Expected (2,50,128), got {out.shape}"
    assert not torch.any(torch.isnan(out))


def test_physics_attention_block():
    from agents.model_architect.physics_block_library import PhysicsAttentionBlock
    block = PhysicsAttentionBlock(hidden_dim=64, n_heads=4, n_slices=8)
    x = torch.randn(2, 100, 64)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.any(torch.isnan(out))


def test_fourier_layer():
    from agents.model_architect.physics_block_library import FourierLayer
    layer = FourierLayer(hidden_dim=32, n_modes=8)
    x = torch.randn(2, 32, 16, 16)
    out = layer(x)
    assert out.shape == x.shape
    assert not torch.any(torch.isnan(out))


def test_graph_conv_block():
    from agents.model_architect.physics_block_library import GraphConvBlock
    block = GraphConvBlock(hidden_dim=64, edge_dim=4)
    N = 50
    E = 200
    node_feats  = torch.randn(N, 64)
    edge_index  = torch.randint(0, N, (2, E))
    edge_feats  = torch.randn(E, 4)
    out = block(node_feats, edge_index, edge_feats)
    assert out.shape == node_feats.shape
    assert not torch.any(torch.isnan(out))


def test_bc_encoder():
    from agents.model_architect.physics_block_library import BoundaryConditionEncoder
    enc = BoundaryConditionEncoder(hidden_dim=64)
    bc_types  = torch.randint(0, 6, (2, 50))
    bc_values = torch.randn(2, 50, 4)
    out = enc(bc_types, bc_values)
    assert out.shape == (2, 50, 64)
    assert not torch.any(torch.isnan(out))
