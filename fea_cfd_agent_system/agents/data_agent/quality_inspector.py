"""Quality inspector — checks mesh and field quality metrics."""

import numpy as np
from loguru import logger


class QualityInspector:
    """Checks simulation data quality and flags issues."""

    def inspect(self, case: dict) -> list:
        """Return list of quality issues found in the case."""
        issues = []

        # NaN/Inf check
        for name, data in case.get("fields", {}).items():
            if isinstance(data, np.ndarray):
                if np.any(np.isnan(data)):
                    issues.append(f"NaN values in field '{name}'")
                if np.any(np.isinf(data)):
                    issues.append(f"Inf values in field '{name}'")

        # Mesh quality
        mesh_quality = case.get("mesh_quality", {})
        skewness = mesh_quality.get("skewness_max", 0.0)
        if skewness > 0.95:
            issues.append(f"High mesh skewness: {skewness:.3f}")

        quality_min = mesh_quality.get("quality_min", 1.0)
        if quality_min < 0.01:
            issues.append(f"Very low cell quality: {quality_min:.4f}")

        # Node count
        n_nodes = case.get("n_nodes", 0)
        if n_nodes < 10:
            issues.append(f"Too few nodes: {n_nodes}")

        return issues
