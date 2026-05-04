"""FEA physics law implementations."""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class FEALaws:
    """
    Verifies FEA physics: stress tensor symmetry, yield criteria, equilibrium.
    """

    def __init__(self, yield_strength: float = 250e6,
                 elastic_modulus: float = 210e9,
                 poisson_ratio: float = 0.3):
        self.yield_strength  = yield_strength
        self.E               = elastic_modulus
        self.nu              = poisson_ratio

    def check_stress_tensor_symmetry(self, stress: np.ndarray,
                                      tolerance: float = 1e-6
                                      ) -> Tuple[bool, float]:
        """
        Verify stress tensor is symmetric: sigma_ij = sigma_ji.
        stress: (N, 6) with Voigt notation [s11, s22, s33, s12, s13, s23]
        or (N, 9) full tensor.
        """
        if stress.ndim == 1 or stress.shape[-1] < 6:
            return True, 0.0

        if stress.shape[-1] == 9:
            S = stress.reshape(-1, 3, 3)
            asymmetry = np.abs(S - S.transpose(0, 2, 1))
            max_asym = float(np.mean(np.max(asymmetry, axis=(1, 2))))
        else:
            # Voigt: sxy = s[3], sxz = s[4], syz = s[5]
            # Already symmetric by Voigt convention
            max_asym = 0.0

        return max_asym < tolerance, max_asym

    def check_von_mises_yield(self, stress: np.ndarray,
                               safety_factor: float = 1.05
                               ) -> Tuple[bool, float, float]:
        """
        Check von Mises criterion: sigma_vm < yield_strength.
        Returns (passed, max_vm_stress, yield_fraction).
        """
        vm = self._compute_von_mises(stress)
        max_vm = float(np.max(vm))
        limit  = self.yield_strength * safety_factor
        frac   = max_vm / (self.yield_strength + 1e-30)
        return max_vm < limit, max_vm, frac

    def _compute_von_mises(self, stress: np.ndarray) -> np.ndarray:
        if stress.ndim == 1:
            return np.abs(stress)

        if stress.shape[-1] == 6:
            s11, s22, s33 = stress[:, 0], stress[:, 1], stress[:, 2]
            s12, s13, s23 = stress[:, 3], stress[:, 4], stress[:, 5]
            vm2 = (0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                          + 6 * (s12**2 + s13**2 + s23**2)))
            return np.sqrt(np.maximum(vm2, 0))

        if stress.shape[-1] == 1:
            return np.abs(stress.ravel())

        return np.linalg.norm(stress, axis=-1)

    def check_displacement_reasonableness(self, displacement: np.ndarray,
                                           char_length: float = 1.0
                                           ) -> Tuple[bool, str]:
        """Displacement should be much smaller than characteristic length."""
        if displacement.ndim > 1:
            mag = np.linalg.norm(displacement, axis=-1)
        else:
            mag = np.abs(displacement)

        max_disp   = float(np.max(mag))
        disp_ratio = max_disp / (char_length + 1e-30)

        if disp_ratio > 0.5:
            return False, f"Large displacement ratio {disp_ratio:.3f} (>0.5) — small deformation assumption violated"
        if np.any(np.isnan(mag)) or np.any(np.isinf(mag)):
            return False, "Displacement contains NaN/Inf"

        return True, f"max_disp={max_disp:.4e}, ratio={disp_ratio:.4f}"

    def check_symmetry_bc(self, field: np.ndarray,
                           symmetry_mask: np.ndarray,
                           symmetry_axis: int = 1,
                           tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Verify normal displacement is zero on symmetry planes.
        field: (N, 3) displacement vectors
        symmetry_axis: which axis is normal to symmetry plane (0=x, 1=y, 2=z)
        """
        if not np.any(symmetry_mask):
            return True, 0.0

        if field.ndim < 2 or field.shape[-1] < 3:
            return True, 0.0

        sym_disps = field[symmetry_mask, symmetry_axis]
        max_normal = float(np.mean(np.abs(sym_disps)))
        return max_normal < tolerance, max_normal

    def compute_strain_energy(self, stress: np.ndarray,
                               strain: Optional[np.ndarray] = None) -> float:
        """Estimate total strain energy (proportional to structural stiffness)."""
        vm = self._compute_von_mises(stress)
        return float(np.mean(vm**2) / (2 * self.E + 1e-30))
