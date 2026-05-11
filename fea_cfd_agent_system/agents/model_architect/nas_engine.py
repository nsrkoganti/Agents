"""
Neural Architecture Search Engine.
Uses Optuna to search over the architecture space.
Finds optimal: hidden_dim, n_layers, n_slices, n_heads, dropout.
"""

import copy
import optuna
import torch
from loguru import logger
from agents.model_architect.architecture_dna import (
    ArchitectureDNA, ArchitectureBlock, BlockType
)


class NASEngine:
    """
    Bayesian optimization over architecture hyperparameters AND block types.
    For fixed templates: searches sizes/counts only.
    For novel architectures (from_llm_json): also searches block type sequences.
    """

    # Block types eligible for NAS block-type search (excludes utility layers)
    SEARCHABLE_CORE_BLOCKS = [
        BlockType.PHYSICS_ATTN,
        BlockType.FOURIER,
        BlockType.GRAPH_CONV,
        BlockType.MAMBA_BLOCK,
        BlockType.CONV_NEXT_BLOCK,
        BlockType.CROSS_ATTENTION,
    ]

    def __init__(self, config: dict):
        self.config = config
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def refine_dna(self, dna: ArchitectureDNA,
                    problem_card, n_trials: int = 20) -> ArchitectureDNA:
        """Search for better hyperparameters (and optionally block types) for the given DNA."""
        logger.info(f"NAS: searching architecture space ({n_trials} trials)")

        # Detect if this is a novel / LLM-designed DNA that allows block-type search
        is_novel = dna.generation >= 2

        def objective(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            n_layers   = trial.suggest_int("n_layers", 2, 12)
            dropout    = trial.suggest_float("dropout", 0.0, 0.3)
            n_slices   = trial.suggest_categorical("n_slices", [8, 16, 32, 64]) \
                         if any(b.block_type == BlockType.PHYSICS_ATTN for b in dna.core_blocks) else 16
            n_heads    = trial.suggest_categorical("n_heads", [4, 8, 16]) \
                         if any(b.block_type in (BlockType.ATTENTION, BlockType.CROSS_ATTENTION)
                                for b in dna.core_blocks) else 8

            # For novel architectures, also sample core block types
            if is_novel and dna.core_blocks:
                searchable = [bt.value for bt in self.SEARCHABLE_CORE_BLOCKS]
                core_type_0 = trial.suggest_categorical("core_type_0", searchable)
                core_type_1 = trial.suggest_categorical("core_type_1", searchable)
                selected_types = [BlockType(core_type_0), BlockType(core_type_1)]
            else:
                selected_types = None

            trial_dna = self._apply_params(dna, hidden_dim, n_layers, dropout,
                                           n_slices, n_heads, selected_types)
            return self._estimate_score(trial_dna, problem_card)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logger.info(f"NAS best: {best}")

        selected_types = None
        if is_novel and dna.core_blocks:
            selected_types = [
                BlockType(best.get("core_type_0", dna.core_blocks[0].block_type.value)),
                BlockType(best.get("core_type_1", dna.core_blocks[-1].block_type.value)),
            ]

        return self._apply_params(
            dna,
            hidden_dim=best["hidden_dim"],
            n_layers=best["n_layers"],
            dropout=best.get("dropout", 0.1),
            n_slices=best.get("n_slices", 32),
            n_heads=best.get("n_heads", 8),
            block_types=selected_types,
        )

    def _apply_params(self, dna: ArchitectureDNA, hidden_dim: int,
                       n_layers: int, dropout: float,
                       n_slices: int, n_heads: int,
                       block_types: list = None) -> ArchitectureDNA:
        new_dna = copy.deepcopy(dna)
        template_block = new_dna.core_blocks[0] if new_dna.core_blocks else None
        if template_block:
            new_dna.core_blocks = []
            for i in range(n_layers):
                b = copy.deepcopy(template_block)
                # Optionally override block type from NAS search
                if block_types:
                    b.block_type = block_types[i % len(block_types)]
                b.hidden_dim = hidden_dim
                b.dropout    = dropout
                b.n_slices   = n_slices
                b.n_heads    = n_heads
                b.residual   = i > 0
                new_dna.core_blocks.append(b)

        for b in new_dna.input_processing + new_dna.output_processing:
            if b.block_type != BlockType.LINEAR or b == new_dna.output_processing[-1]:
                continue
            b.hidden_dim = hidden_dim

        return new_dna

    def _estimate_score(self, dna: ArchitectureDNA, problem_card) -> float:
        """Heuristic score for architecture quality."""
        total_params  = sum(b.hidden_dim * b.hidden_dim for b in dna.core_blocks)
        physics_bonus = 2.0 if dna.has_physics_loss else 0.0
        mesh_bonus    = 1.0 if dna.mesh_type == "any" else 0.5
        size_penalty  = max(0, (total_params - 5_000_000) / 1_000_000)
        return physics_bonus + mesh_bonus - size_penalty
