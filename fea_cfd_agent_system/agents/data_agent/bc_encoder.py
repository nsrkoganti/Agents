"""Boundary condition encoder — identifies and encodes BC types for each mesh node."""

import numpy as np
from loguru import logger


BC_TYPE_MAP = {
    "inlet":    0,
    "outlet":   1,
    "wall":     2,
    "symmetry": 3,
    "interior": 4,
    "unknown":  5,
}


class BCEncoder:
    """Encodes boundary condition information into node-level feature arrays."""

    def encode(self, case: dict) -> np.ndarray:
        """Encode BC type per node as integer array."""
        n_nodes = case.get("n_nodes", 0)
        bc_ids = np.full(n_nodes, BC_TYPE_MAP["interior"], dtype=np.int32)

        boundary_info = case.get("boundary_info", {})
        for bc_type, idx_key in [
            ("wall",     "wall_node_indices"),
            ("inlet",    "inlet_node_indices"),
            ("outlet",   "outlet_node_indices"),
            ("symmetry", "symmetry_node_indices"),
        ]:
            indices = boundary_info.get(idx_key, [])
            if len(indices) > 0:
                bc_ids[indices] = BC_TYPE_MAP[bc_type]

        return bc_ids

    def encode_bc_values(self, case: dict, n_nodes: int) -> np.ndarray:
        """Encode BC value magnitudes per node as float array (4 channels)."""
        bc_vals = np.zeros((n_nodes, 4), dtype=np.float32)
        bcs = case.get("boundary_conditions", {})
        inlet_bc = bcs.get("inlet", {})
        u_inlet = inlet_bc.get("U", 1.0)
        inlet_idx = case.get("boundary_info", {}).get("inlet_node_indices", [])
        if len(inlet_idx) > 0:
            bc_vals[inlet_idx, 0] = float(u_inlet)
        return bc_vals
