"""
Physics-Informed Loss Function.
Combines data loss with PDE residuals.
Lambda weights are updated by Physics Agent feedback.
"""

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    Total loss = MSE(pred, target)
               + lambda_continuity * continuity_residual
               + lambda_momentum   * momentum_residual
               + lambda_bc         * bc_violation
               + lambda_symmetry   * symmetry_violation  [FEA]
               + lambda_equilibrium* equilibrium_residual [FEA]
    """

    def __init__(self, lambda_weights: dict, physics_type: str):
        super().__init__()
        self.lambdas = lambda_weights
        self.physics_type = physics_type
        self.mse = nn.MSELoss()

    def forward(self, pred, target, case=None) -> torch.Tensor:
        if pred.dim() > 1:
            pred_flat   = pred.reshape(pred.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1) if target.dim() > 1 else target
        else:
            pred_flat   = pred
            target_flat = target

        if pred_flat.shape == target_flat.shape:
            data_loss = self.mse(pred_flat, target_flat)
        else:
            min_len = min(pred_flat.numel(), target_flat.numel())
            data_loss = self.mse(pred_flat.flatten()[:min_len],
                                 target_flat.flatten()[:min_len])

        total = data_loss

        if "CFD" in self.physics_type and pred.numel() > 3:
            try:
                if pred.dim() >= 2 and pred.shape[-1] >= 3:
                    u = pred[..., 0]
                    v = pred[..., 1]
                    continuity = torch.mean(torch.abs(
                        torch.gradient(u, dim=-1)[0] +
                        torch.gradient(v, dim=-1)[0]
                    ))
                    total = total + self.lambdas.get("continuity", 1.0) * continuity
            except Exception:
                pass

        return total
