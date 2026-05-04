"""Mesh reading utilities for various simulation file formats."""

import numpy as np
from pathlib import Path
from loguru import logger


class MeshReader:
    """Reads mesh topology and connectivity from simulation files."""

    def read_connectivity(self, filepath: str) -> dict:
        """Read mesh connectivity (cell-node mapping)."""
        try:
            import pyvista as pv
            mesh = pv.read(filepath)
            return {
                "n_cells": mesh.n_cells,
                "n_points": mesh.n_points,
                "cell_types": mesh.celltypes.tolist() if hasattr(mesh, 'celltypes') else [],
            }
        except Exception as e:
            logger.warning(f"Mesh connectivity read failed: {e}")
            return {}

    def compute_cell_centers(self, mesh) -> np.ndarray:
        """Compute cell center coordinates."""
        try:
            centers = mesh.cell_centers()
            return np.array(centers.points, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)
