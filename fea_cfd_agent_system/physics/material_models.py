"""Material model constraints for FEA validation."""

import numpy as np
from typing import Tuple, Dict


class MaterialModels:
    """Validates material behavior in FEA predictions."""

    def __init__(self, yield_strength: float = 250e6,
                 ultimate_strength: float = 400e6,
                 density: float = 7850.0):
        self.yield_strength    = yield_strength
        self.ultimate_strength = ultimate_strength
        self.density           = density

    def check_damage_variable(self, damage: np.ndarray) -> Tuple[bool, float]:
        """Damage variable D must be in [0, 1]."""
        max_d  = float(np.max(damage))
        min_d  = float(np.min(damage))
        valid  = 0.0 <= min_d and max_d <= 1.0
        return valid, max_d

    def check_elastic_range(self, stress: np.ndarray,
                             strain: np.ndarray,
                             E: float = 210e9) -> Tuple[bool, float]:
        """
        In elastic range, sigma ≈ E * epsilon.
        Check correlation between stress and strain magnitudes.
        """
        if stress.ndim > 1:
            s_mag = np.linalg.norm(stress, axis=-1)
        else:
            s_mag = np.abs(stress)

        if strain.ndim > 1:
            e_mag = np.linalg.norm(strain, axis=-1)
        else:
            e_mag = np.abs(strain)

        # For nodes below yield, sigma ≈ E*epsilon
        below_yield = s_mag < self.yield_strength
        if not np.any(below_yield):
            return True, 0.0

        ratio = s_mag[below_yield] / (E * e_mag[below_yield] + 1e-30)
        mean_ratio = float(np.mean(ratio))
        valid = 0.5 < mean_ratio < 2.0
        return valid, mean_ratio

    def check_material_bounds(self, fields: Dict[str, np.ndarray]) -> Dict[str, bool]:
        results = {}

        if "damage" in fields:
            ok, _ = self.check_damage_variable(fields["damage"])
            results["damage_valid"] = ok

        if "temperature" in fields:
            T = fields["temperature"]
            results["temperature_positive"] = bool(np.all(T > 0))
            results["temperature_reasonable"] = bool(np.max(T) < 5000)

        if "density" in fields:
            rho = fields["density"]
            results["density_positive"] = bool(np.all(rho > 0))

        return results
