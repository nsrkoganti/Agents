"""
Mixture of Experts Surrogate Model — per-node spatial gating.

At every mesh node the gating network assigns soft weights to each expert.
Different regions route to different specialists:
  - Smooth interior      → FNO / Transolver
  - Stress concentrations → GNN / EAGNN
  - Boundary nodes       → PINN / physics-constrained models

Shape contract (shared with all surrogates):
  forward(x: (B, N, in_dim)) -> (B, N, out_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NodeGatingNetwork(nn.Module):
    """
    Learns per-node expert weights from node features.
    Output: softmax probabilities over experts for every node.
    """

    def __init__(self, in_dim: int, n_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim) -> gates: (B, N, n_experts)"""
        return F.softmax(self.net(x), dim=-1)


class MixtureOfExpertsSurrogate(nn.Module):
    """
    Soft per-node Mixture of Experts for FEA/CFD field prediction.

    output[b, n] = sum_k( gate[b, n, k] * expert_k(x)[b, n] )

    Training happens in three phases (managed by TrainerAgent.train_moe):
      1. Pre-train each expert independently
      2. Train gating network with experts frozen
      3. Optional full end-to-end fine-tune at low LR
    """

    def __init__(
        self,
        experts: List[nn.Module],
        in_dim: int,
        out_dim: int,
        gate_hidden_dim: int = 64,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        assert len(experts) >= 2, "Need at least 2 experts"
        self.experts    = nn.ModuleList(experts)
        self.n_experts  = len(experts)
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.load_balance_weight = load_balance_weight

        self.gate = NodeGatingNetwork(in_dim, self.n_experts, gate_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim)
        Returns: (B, N, out_dim)
        """
        gates = self.gate(x)                             # (B, N, K)

        expert_outs = []
        for expert in self.experts:
            try:
                out = expert(x)                          # (B, N, out_dim)
            except Exception:
                # Expert may expect different calling convention — skip gracefully
                out = torch.zeros(x.shape[0], x.shape[1], self.out_dim,
                                  device=x.device, dtype=x.dtype)
            expert_outs.append(out)

        stacked = torch.stack(expert_outs, dim=2)        # (B, N, K, out_dim)
        weighted = (gates.unsqueeze(-1) * stacked).sum(dim=2)  # (B, N, out_dim)
        return weighted

    def compute_physics_loss(self, pred: Optional[torch.Tensor] = None,
                              coords: Optional[torch.Tensor] = None,
                              x: Optional[torch.Tensor] = None) -> dict:
        """
        Aggregate physics losses from all experts (weighted by mean gate prob)
        plus a load-balance loss to prevent gate collapse.
        """
        losses: dict = {}

        if x is not None:
            gates = self.gate(x)                         # (B, N, K)
            mean_gates = gates.mean(dim=(0, 1))          # (K,) — mean load per expert

            # Collect per-expert physics losses
            for k, expert in enumerate(self.experts):
                if hasattr(expert, "compute_physics_loss"):
                    try:
                        expert_losses = expert.compute_physics_loss(pred, coords)
                        for name, val in expert_losses.items():
                            key = f"expert_{k}_{name}"
                            losses[key] = val * mean_gates[k]
                    except Exception:
                        pass

            # Load-balance loss: penalise variance in expert utilisation
            # Encourages all experts to be used roughly equally
            load_balance = self.load_balance_weight * mean_gates.var()
            losses["load_balance"] = load_balance

        return losses

    def get_expert_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-node expert gate weights for visualisation.
        x: (B, N, in_dim) -> (B, N, n_experts)
        """
        with torch.no_grad():
            return self.gate(x)

    def freeze_experts(self):
        """Freeze all expert parameters (for gating-only training phase)."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad_(False)

    def unfreeze_experts(self):
        """Unfreeze all expert parameters (for end-to-end fine-tuning)."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad_(True)

    def gate_parameters(self):
        """Return only gating network parameters."""
        return self.gate.parameters()
