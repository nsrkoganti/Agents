"""Training loop utilities — early stopping, gradient clipping, LR scheduling."""

import torch
from loguru import logger


class EarlyStopping:
    """Track validation loss and signal when to stop training."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_optimizer(model: torch.nn.Module, lr: float = 1e-3,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Build Adam optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, max_epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Build cosine annealing scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
