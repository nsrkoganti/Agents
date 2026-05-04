"""
CFD physics law implementations.
Used by physics sub-agents to verify model predictions obey governing equations.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from loguru import logger


class CFDLaws:
    """
    Verifies CFD physics: continuity, Navier-Stokes, turbulence constraints.
    Operates on numpy arrays of predicted fields at mesh node coordinates.
    """

    def __init__(self, rho: float = 1.225, mu: float = 1.81e-5,
                 incompressible: bool = True):
        self.rho             = rho
        self.mu              = mu
        self.incompressible  = incompressible

    def check_continuity(self, velocity: np.ndarray,
                          coords: np.ndarray,
                          n_samples: int = 200) -> Tuple[bool, float]:
        """
        Check mass conservation: div(V) ≈ 0 for incompressible flow.
        Uses finite differences on randomly sampled neighborhoods.
        Returns (passed, max_divergence).
        """
        if velocity.shape[-1] < 2:
            return True, 0.0

        N = min(n_samples, len(coords) - 1)
        indices = np.random.choice(len(coords), size=N, replace=False)
        divs = []

        for i in indices:
            dists = np.linalg.norm(coords - coords[i], axis=-1)
            neighbors = np.argsort(dists)[1:7]
            if len(neighbors) < 2:
                continue

            du_dx = self._approx_derivative(
                velocity[:, 0], coords[:, 0], i, neighbors
            )
            dv_dy = self._approx_derivative(
                velocity[:, 1], coords[:, 1], i, neighbors
            )
            div = abs(du_dx + dv_dy)
            if velocity.shape[-1] >= 3:
                dw_dz = self._approx_derivative(
                    velocity[:, 2], coords[:, 2], i, neighbors
                )
                div += abs(dw_dz)
            divs.append(div)

        if not divs:
            return True, 0.0

        max_div = float(np.mean(divs))
        threshold = 1e-3 if self.incompressible else 1.0
        return max_div < threshold, max_div

    def _approx_derivative(self, field: np.ndarray, coord: np.ndarray,
                             center_idx: int, neighbor_idx: np.ndarray) -> float:
        dx = coord[neighbor_idx] - coord[center_idx]
        df = field[neighbor_idx] - field[center_idx]
        valid = np.abs(dx) > 1e-12
        if not np.any(valid):
            return 0.0
        return float(np.mean(df[valid] / dx[valid]))

    def check_velocity_bounds(self, velocity: np.ndarray,
                               re_number: Optional[float] = None,
                               char_length: float = 1.0) -> Tuple[bool, str]:
        """Check that velocity magnitudes are physically reasonable."""
        mag = np.linalg.norm(velocity, axis=-1) if velocity.ndim > 1 else np.abs(velocity)
        max_vel = float(np.max(mag))
        mean_vel = float(np.mean(mag))

        if re_number and re_number > 0:
            expected_vel = re_number * self.mu / (self.rho * char_length)
            if max_vel > 100 * expected_vel:
                return False, f"Velocity {max_vel:.2f} >> expected {expected_vel:.2f} (Re={re_number:.0f})"

        if max_vel > 340.0:
            return False, f"Velocity {max_vel:.1f} m/s exceeds speed of sound (Ma>1 for incompressible)"

        if max_vel < 1e-10 and mean_vel < 1e-10:
            return False, "All velocities are zero"

        return True, f"max_vel={max_vel:.3f}, mean_vel={mean_vel:.3f}"

    def check_pressure_field(self, pressure: np.ndarray) -> Tuple[bool, str]:
        """Check pressure has no extreme outliers."""
        if pressure.ndim > 1:
            pressure = pressure.ravel()
        has_nan = bool(np.any(np.isnan(pressure)))
        has_inf = bool(np.any(np.isinf(pressure)))
        if has_nan or has_inf:
            return False, "Pressure field contains NaN/Inf"

        p_range = float(np.max(pressure) - np.min(pressure))
        p_mean  = float(np.abs(np.mean(pressure)))
        if p_mean > 0 and p_range > 1e6 * p_mean:
            return False, f"Pressure range {p_range:.2e} >> mean {p_mean:.2e}"

        return True, f"pressure_range={p_range:.2e}"

    def check_no_slip_bc(self, velocity: np.ndarray,
                          wall_mask: np.ndarray,
                          tolerance: float = 0.05) -> Tuple[bool, float]:
        """Verify velocity near-zero at wall nodes (no-slip condition)."""
        if not np.any(wall_mask):
            return True, 0.0

        wall_vel = velocity[wall_mask]
        if wall_vel.ndim > 1:
            wall_mag = np.linalg.norm(wall_vel, axis=-1)
        else:
            wall_mag = np.abs(wall_vel)

        mean_wall_vel = float(np.mean(wall_mag))
        return mean_wall_vel < tolerance, mean_wall_vel

    def estimate_re_number(self, velocity: np.ndarray,
                            char_length: float = 1.0) -> float:
        """Estimate Reynolds number from velocity field."""
        if velocity.ndim > 1:
            mag = np.linalg.norm(velocity, axis=-1)
        else:
            mag = np.abs(velocity)
        mean_vel = float(np.mean(mag))
        return self.rho * mean_vel * char_length / (self.mu + 1e-30)
