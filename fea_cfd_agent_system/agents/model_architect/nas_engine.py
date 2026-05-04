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
    Bayesian optimization over architecture hyperparameters.
    Doesn't change the block types — only sizes and counts.
    """

    def __init__(self, config: dict):
        self.config = config
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def refine_dna(self, dna: ArchitectureDNA,
                    problem_card, n_trials: int = 20) -> ArchitectureDNA:
        """Search for better hyperparameters for the given DNA template."""
        logger.info(f"NAS: searching architecture space ({n_trials} trials)")

        def objective(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            n_layers   = trial.suggest_int("n_layers", 2, 12)
            dropout    = trial.suggest_float("dropout", 0.0, 0.3)
            n_slices   = trial.suggest_categorical("n_slices", [8, 16, 32, 64]) \
                         if any(b.block_type == BlockType.PHYSICS_ATTN for b in dna.core_blocks) else 16
            n_heads    = trial.suggest_categorical("n_heads", [4, 8, 16]) \
                         if any(b.block_type == BlockType.ATTENTION for b in dna.core_blocks) else 8

            trial_dna = self._apply_params(dna, hidden_dim, n_layers, dropout, n_slices, n_heads)
            return self._estimate_score(trial_dna, problem_card)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logger.info(f"NAS best: {best}")

        return self._apply_params(
            dna,
            hidden_dim=best["hidden_dim"],
            n_layers=best["n_layers"],
            dropout=best.get("dropout", 0.1),
            n_slices=best.get("n_slices", 32),
            n_heads=best.get("n_heads", 8),
        )

    def _apply_params(self, dna: ArchitectureDNA, hidden_dim: int,
                       n_layers: int, dropout: float,
                       n_slices: int, n_heads: int) -> ArchitectureDNA:
        new_dna = copy.deepcopy(dna)
        template_block = new_dna.core_blocks[0] if new_dna.core_blocks else None
        if template_block:
            new_dna.core_blocks = []
            for i in range(n_layers):
                b = copy.deepcopy(template_block)
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
