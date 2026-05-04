"""Field extraction utilities — extracts physical fields from simulation data."""

import numpy as np
from loguru import logger


class FieldExtractor:
    """Extracts and normalizes physical fields from raw simulation data."""

    FIELD_ALIASES = {
        "velocity": ["U", "Velocity", "vel", "u", "v_mag"],
        "pressure": ["P", "Pressure", "p"],
        "k":        ["TurbulentKineticEnergy", "TKE", "tke", "k"],
        "omega":    ["SpecificDissipationRate", "omega", "w"],
        "epsilon":  ["TurbulentDissipationRate", "epsilon", "eps"],
    }

    def extract(self, raw_dict: dict) -> dict:
        """Extract standard fields from raw data dict."""
        fields = raw_dict.get("fields", {})
        standardized = {}
        for standard_name, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias in fields:
                    standardized[standard_name] = fields[alias]
                    break
        return {**fields, **standardized}

    def compute_derived(self, fields: dict) -> dict:
        """Compute derived quantities (velocity magnitude, etc.)."""
        derived = {}
        velocity = fields.get("velocity")
        if velocity is not None and velocity.ndim > 1:
            derived["velocity_magnitude"] = np.linalg.norm(velocity, axis=-1)
        return {**fields, **derived}
