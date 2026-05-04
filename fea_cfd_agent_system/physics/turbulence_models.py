"""Turbulence model constraints and validation."""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class TurbulenceModels:
    """Validates turbulence field predictions against physical constraints."""

    REGIMES = {
        "laminar":          (0, 2300),
        "transitional":     (2300, 4000),
        "turbulent":        (4000, 1e8),
        "highly_turbulent": (1e8, np.inf),
    }

    def classify_regime(self, re: float) -> str:
        for name, (lo, hi) in self.REGIMES.items():
            if lo <= re < hi:
                return name
        return "unknown"

    def check_tke_positive(self, k: np.ndarray) -> Tuple[bool, float]:
        """TKE must be non-negative everywhere."""
        n_negative = int(np.sum(k < 0))
        frac_neg = n_negative / (len(k) + 1e-10)
        return frac_neg < 0.01, frac_neg

    def check_omega_positive(self, omega: np.ndarray) -> Tuple[bool, float]:
        """Specific dissipation rate must be strictly positive."""
        n_nonpos = int(np.sum(omega <= 0))
        frac_nonpos = n_nonpos / (len(omega) + 1e-10)
        return frac_nonpos < 0.01, frac_nonpos

    def check_turbulence_intensity(self, k: np.ndarray,
                                    velocity: np.ndarray) -> Tuple[bool, float]:
        """
        Turbulence intensity I = sqrt(2k/3) / |U| should be in [0.1%, 50%].
        """
        if velocity.ndim > 1:
            u_mag = np.linalg.norm(velocity, axis=-1)
        else:
            u_mag = np.abs(velocity)

        u_ref = float(np.mean(u_mag)) + 1e-8
        tke   = np.maximum(k, 0)
        I     = np.sqrt(2.0 * tke / 3.0) / u_ref
        mean_I = float(np.mean(I))
        return 0.001 < mean_I < 0.5, mean_I

    def check_y_plus(self, y_plus: Optional[np.ndarray],
                      wall_treatment: str = "wall_function"
                      ) -> Tuple[bool, str]:
        """
        y+ validity:
        - Wall functions: y+ in [30, 300]
        - Low-Re (resolved): y+ < 1
        """
        if y_plus is None:
            return True, "y+ not available"

        mean_yp = float(np.mean(y_plus))
        max_yp  = float(np.max(y_plus))

        if wall_treatment == "wall_function":
            valid = 30 <= mean_yp <= 300
            msg = f"mean_y+={mean_yp:.1f} (target 30–300)"
        else:
            valid = max_yp < 5.0
            msg = f"max_y+={max_yp:.2f} (target <1 for resolved BL)"

        return valid, msg

    def check_k_omega_sst_bounds(self, k: np.ndarray,
                                   omega: np.ndarray) -> Tuple[bool, dict]:
        """Check k-omega SST specific constraints."""
        issues = {}
        k_pos, k_frac = self.check_tke_positive(k)
        o_pos, o_frac = self.check_omega_positive(omega)

        if not k_pos:
            issues["negative_k"] = f"{k_frac:.1%} nodes have k<0"
        if not o_pos:
            issues["nonpositive_omega"] = f"{o_frac:.1%} nodes have omega<=0"

        mu_t  = k / (np.maximum(omega, 1e-10))
        if float(np.max(mu_t)) > 1e4:
            issues["excessive_nut"] = f"nut_max={float(np.max(mu_t)):.2e} (physically implausible)"

        return len(issues) == 0, issues
