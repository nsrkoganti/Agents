"""
Auto-Configurator — automatically sizes the model architecture
based on data size, problem type, and compute budget.
"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class SimpleMLPSurrogate(nn.Module):
    """
    Fallback MLP surrogate — always works, partial accuracy.
    Used when specialized models are not available.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class AutoConfigurator:
    """
    Automatically selects model architecture and training config
    based on the Problem Card and dataset statistics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("system", {}).get("device", "cpu")

    def build_model(self, model_name: str, problem_card: ProblemCard,
                    dataset: dict, config_hints: str = "",
                    use_optuna: bool = False,
                    custom_model=None) -> tuple:
        """Build and return (model, config_dict)."""
        n_data  = problem_card.data_size
        n_nodes = problem_card.n_nodes or 1000

        if n_data < 100:
            tier = "small"
            hidden_dims = [64, 64]
            dropout = 0.4
            lr = 5e-4
            max_epochs = 500
        elif n_data < 500:
            tier = "medium"
            hidden_dims = [128, 256, 128]
            dropout = 0.2
            lr = 1e-3
            max_epochs = 1000
        else:
            tier = "large"
            hidden_dims = [256, 512, 512, 256]
            dropout = 0.1
            lr = 1e-3
            max_epochs = 2000

        cases = dataset.get("cases", [])
        if cases:
            sample = cases[0]
            coords = sample.get("node_coords")
            input_dim = coords.shape[-1] if coords is not None else 3
            fields = sample.get("fields", {})
            output_dim = sum(
                v.shape[-1] if v.ndim > 1 else 1
                for v in fields.values()
                if hasattr(v, "shape")
            )
            output_dim = max(output_dim, 1)
        else:
            input_dim  = 3
            output_dim = 4

        config_used = {
            "tier":        tier,
            "hidden_dims": hidden_dims,
            "dropout":     dropout,
            "lr":          lr,
            "max_epochs":  max_epochs,
            "input_dim":   input_dim,
            "output_dim":  output_dim,
        }

        # Use pre-built custom model if provided
        if custom_model is not None:
            model = custom_model
        else:
            model = self._try_build_specialized(model_name, input_dim, output_dim, config_used)

        if model is None:
            logger.warning(f"{model_name} not available — using MLP fallback")
            model = SimpleMLPSurrogate(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        model = model.to(self.device)
        logger.info(
            f"Built {model_name} ({tier} tier): "
            f"in={input_dim}, out={output_dim}, "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )
        return model, config_used

    def _try_build_specialized(self, model_name: str, input_dim: int,
                                output_dim: int, config: dict):
        """Try to build the specific requested model."""
        try:
            if model_name in ("FNO", "F-FNO"):
                from neuraloperator.models import FNO
                return FNO(
                    n_modes=(16, 16),
                    hidden_channels=64,
                    in_channels=input_dim,
                    out_channels=output_dim,
                    n_layers=4,
                )
            elif model_name == "GINO":
                from neuraloperator.models import GINO
                return GINO(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    hidden_channels=64,
                    n_modes=(16, 16, 16),
                    gno_radius=0.033,
                )
            elif model_name == "DeepONet":
                return None
            elif model_name == "MeshGraphNet":
                from physicsnemo.models import MeshGraphNet
                return MeshGraphNet(
                    input_dim_nodes=input_dim,
                    input_dim_edges=4,
                    output_dim=output_dim,
                    processor_size=15,
                    hidden_dim_node_encoder=128,
                    hidden_dim_edge_encoder=128,
                    hidden_dim_node_decoder=128,
                )
            else:
                return None
        except ImportError as e:
            logger.debug(f"Could not import {model_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to build {model_name}: {e}")
            return None
