"""
Architecture DNA — defines every possible building block.
The architect picks and combines these to design new models.
Think of it as a genome for neural architectures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class BlockType(str, Enum):
    LINEAR        = "linear"
    CONV1D        = "conv1d"
    CONV3D        = "conv3d"
    ATTENTION     = "attention"
    PHYSICS_ATTN  = "physics_attention"
    FOURIER       = "fourier_layer"
    GRAPH_CONV    = "graph_conv"
    GRAPH_ATTN    = "graph_attention"
    LAYER_NORM    = "layer_norm"
    BATCH_NORM    = "batch_norm"
    GELU          = "gelu"
    TANH          = "tanh"
    SILU          = "silu"
    PHYSICS_LOSS  = "physics_loss_layer"
    BC_ENCODER    = "bc_encoder"
    COORD_EMBED   = "coord_embed"
    PHYSICS_STATE = "physics_state_slice"
    RESIDUAL      = "residual_connection"
    HIGHWAY       = "highway_gate"
    GLOBAL_POOL   = "global_pooling"
    SLICE_POOL    = "slice_pooling"
    # Novel block types for LLM-designed architectures
    MAMBA_BLOCK      = "mamba_block"       # selective state-space, O(N) complexity
    CONV_NEXT_BLOCK  = "conv_next_block"   # depthwise conv + LayerNorm, modern CNN
    CROSS_ATTENTION  = "cross_attention"   # cross-attn between mesh nodes & query points
    SPECTRAL_NORM    = "spectral_norm"     # equivariant spectral layer


@dataclass
class ArchitectureBlock:
    block_type: BlockType
    hidden_dim: int   = 128
    n_heads:    int   = 8
    n_modes:    int   = 16
    n_slices:   int   = 32
    dropout:    float = 0.1
    activation: str   = "gelu"
    residual:   bool  = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureDNA:
    """
    Complete genetic description of a neural architecture.
    Can be serialized to JSON, mutated, and crossed with others.
    """
    name:             str
    family:           str
    input_processing: List[ArchitectureBlock] = field(default_factory=list)
    core_blocks:      List[ArchitectureBlock] = field(default_factory=list)
    output_processing:List[ArchitectureBlock] = field(default_factory=list)
    has_physics_loss:  bool       = True
    physics_loss_types:List[str]  = field(default_factory=list)
    mesh_type:         str        = "any"
    designed_for:      str        = ""
    generation:        int        = 1
    parent_names:      List[str]  = field(default_factory=list)

    def to_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ArchitectureDNA":
        def rebuild_block(b):
            b["block_type"] = BlockType(b["block_type"])
            return ArchitectureBlock(**b)
        d["input_processing"]  = [rebuild_block(b) for b in d.get("input_processing", [])]
        d["core_blocks"]       = [rebuild_block(b) for b in d.get("core_blocks", [])]
        d["output_processing"] = [rebuild_block(b) for b in d.get("output_processing", [])]
        return cls(**d)

    @classmethod
    def from_llm_json(cls, json_dict: Dict) -> "ArchitectureDNA":
        """
        Build ArchitectureDNA from a free-form JSON dict produced by the LLM.
        Validates block types against the BlockType enum.
        Expected JSON shape:
          {
            "name": "CustomNovelNet",
            "family": "hybrid",
            "designed_for": "FEA_static_nonlinear",
            "mesh_type": "any",
            "has_physics_loss": true,
            "physics_loss_types": ["equilibrium", "bc"],
            "input_blocks": [{"type": "coord_embed", "hidden_dim": 128}],
            "core_blocks":  [{"type": "mamba_block", "hidden_dim": 256}, ...],
            "output_blocks": [{"type": "linear", "hidden_dim": 4}]
          }
        """
        def _parse_block(b: Dict) -> ArchitectureBlock:
            raw_type = b.get("type", b.get("block_type", "linear"))
            try:
                bt = BlockType(raw_type)
            except ValueError:
                bt = BlockType.LINEAR
            return ArchitectureBlock(
                block_type=bt,
                hidden_dim=b.get("hidden_dim", 128),
                n_heads=b.get("n_heads", 8),
                n_modes=b.get("n_modes", 16),
                n_slices=b.get("n_slices", 32),
                dropout=b.get("dropout", 0.1),
                activation=b.get("activation", "gelu"),
                residual=b.get("residual", True),
                kwargs=b.get("kwargs", {}),
            )

        return cls(
            name=json_dict.get("name", "LLMDesignedModel"),
            family=json_dict.get("family", "hybrid"),
            designed_for=json_dict.get("designed_for", ""),
            mesh_type=json_dict.get("mesh_type", "any"),
            has_physics_loss=json_dict.get("has_physics_loss", True),
            physics_loss_types=json_dict.get("physics_loss_types", ["equilibrium", "bc"]),
            input_processing=[_parse_block(b) for b in json_dict.get("input_blocks", [])],
            core_blocks=[_parse_block(b) for b in json_dict.get("core_blocks", [])],
            output_processing=[_parse_block(b) for b in json_dict.get("output_blocks", [])],
            generation=json_dict.get("generation", 1),
            parent_names=json_dict.get("parent_names", []),
        )


# ── Pre-defined DNA templates ─────────────────────────────────

def pinn_dna(hidden_dim: int = 128, n_layers: int = 6,
              physics_types: List[str] = None) -> ArchitectureDNA:
    """Standard Physics-Informed Neural Network DNA."""
    return ArchitectureDNA(
        name="CustomPINN",
        family="pinn",
        input_processing=[
            ArchitectureBlock(BlockType.COORD_EMBED, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.BC_ENCODER,  hidden_dim=hidden_dim),
        ],
        core_blocks=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim,
                              activation="tanh", residual=(i > 1))
            for i in range(n_layers)
        ],
        output_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=4),
        ],
        has_physics_loss=True,
        physics_loss_types=physics_types or ["continuity", "momentum", "bc"],
        mesh_type="any",
        designed_for="CFD_incompressible_turbulent",
    )


def transolver_dna(hidden_dim: int = 256, n_heads: int = 8,
                    n_slices: int = 32, n_layers: int = 8) -> ArchitectureDNA:
    """
    Transolver-style DNA.
    Key innovation: Physics-Attention slices mesh into physical states.
    """
    return ArchitectureDNA(
        name="CustomTransolver",
        family="transformer",
        input_processing=[
            ArchitectureBlock(BlockType.COORD_EMBED, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.BC_ENCODER,  hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LAYER_NORM,  hidden_dim=hidden_dim),
        ],
        core_blocks=[
            ArchitectureBlock(
                BlockType.PHYSICS_ATTN,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_slices=n_slices,
                residual=True,
            )
            for _ in range(n_layers)
        ],
        output_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.GELU,   hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=4),
        ],
        has_physics_loss=False,
        mesh_type="any",
        designed_for="CFD_incompressible_turbulent",
    )


def fno_dna(n_modes: int = 16, hidden_dim: int = 64, n_layers: int = 4) -> ArchitectureDNA:
    """Fourier Neural Operator DNA."""
    return ArchitectureDNA(
        name="CustomFNO",
        family="operator",
        input_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim),
        ],
        core_blocks=[
            ArchitectureBlock(BlockType.FOURIER, hidden_dim=hidden_dim,
                              n_modes=n_modes, residual=True)
            for _ in range(n_layers)
        ],
        output_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.GELU,   hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=4),
        ],
        has_physics_loss=False,
        mesh_type="structured",
        designed_for="CFD_incompressible",
    )


def gnn_dna(hidden_dim: int = 128, n_layers: int = 10) -> ArchitectureDNA:
    """Graph Neural Network DNA (MeshGraphNet style)."""
    return ArchitectureDNA(
        name="CustomGNN",
        family="gnn",
        input_processing=[
            ArchitectureBlock(BlockType.LINEAR,     hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LAYER_NORM, hidden_dim=hidden_dim),
        ],
        core_blocks=[
            ArchitectureBlock(BlockType.GRAPH_CONV, hidden_dim=hidden_dim,
                              residual=True)
            for _ in range(n_layers)
        ],
        output_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.GELU,   hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=4),
        ],
        has_physics_loss=False,
        mesh_type="unstructured",
        designed_for="FEA_static_linear",
    )


def hybrid_transolver_pinn_dna(hidden_dim: int = 256,
                                 n_slices: int = 32,
                                 n_layers: int = 6) -> ArchitectureDNA:
    """
    HYBRID: Transolver + PINN.
    Uses Transolver's physics-state attention for geometry,
    PLUS physics residual loss for training.
    """
    return ArchitectureDNA(
        name="CustomTransolverPINN",
        family="hybrid",
        input_processing=[
            ArchitectureBlock(BlockType.COORD_EMBED, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.BC_ENCODER,  hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LAYER_NORM,  hidden_dim=hidden_dim),
        ],
        core_blocks=[
            ArchitectureBlock(BlockType.PHYSICS_ATTN,
                              hidden_dim=hidden_dim, n_slices=n_slices,
                              n_heads=8, residual=True)
            for _ in range(n_layers)
        ],
        output_processing=[
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.GELU,   hidden_dim=hidden_dim),
            ArchitectureBlock(BlockType.LINEAR, hidden_dim=4),
        ],
        has_physics_loss=True,
        physics_loss_types=["continuity", "momentum", "bc", "energy"],
        mesh_type="any",
        designed_for="CFD_incompressible_turbulent",
        parent_names=["Transolver", "PINN"],
    )
