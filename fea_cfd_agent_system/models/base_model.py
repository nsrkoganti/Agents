"""Base class for all surrogate models in the system."""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from loguru import logger


class BaseSurrogateModel(nn.Module, ABC):
    """
    Base class that all surrogate models must inherit from.
    Enforces a consistent interface for training, inference, and physics loss.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_nodes, input_dim)
        returns: (batch, n_nodes, output_dim)
        """
        ...

    def compute_physics_loss(self, pred: torch.Tensor,
                              coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Override to add physics-informed loss terms."""
        return {}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, coords: np.ndarray) -> np.ndarray:
        """Convenience: numpy in → numpy out."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
            y = self.forward(x)
        return y.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim":  self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "class_name": self.__class__.__name__,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs):
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            input_dim  = checkpoint.get("input_dim",  kwargs.get("input_dim", 3)),
            output_dim = checkpoint.get("output_dim", kwargs.get("output_dim", 4)),
            hidden_dim = checkpoint.get("hidden_dim", kwargs.get("hidden_dim", 128)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
