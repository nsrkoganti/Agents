"""Boundary condition encoder — identifies and encodes FEA BC types per mesh node."""

import numpy as np
from loguru import logger


FEA_BC_TYPES = {
    "fixed":    0,   # All DOF locked (encastre)
    "pinned":   1,   # Translations fixed, rotations free
    "symmetry": 2,   # Normal displacement = 0
    "load":     3,   # Applied force or pressure
    "contact":  4,   # Contact boundary
    "free":     5,   # No constraint
    "thermal":  6,   # Temperature prescribed
    "interior": 7,   # Interior node (not on boundary)
}


class BCEncoder:
    """Encodes boundary condition information into node-level feature arrays."""

    def encode(self, case: dict) -> np.ndarray:
        """Encode FEA BC type per node as integer array."""
        n_nodes = case.get("n_nodes", 0)
        bc_ids  = np.full(n_nodes, FEA_BC_TYPES["interior"], dtype=np.int32)

        boundary_info = case.get("boundary_info", {})
        for bc_type, idx_keys in [
            ("fixed",    ["fixed_node_indices",    "encastre_node_indices"]),
            ("pinned",   ["pinned_node_indices"]),
            ("symmetry", ["symmetry_node_indices"]),
            ("load",     ["load_node_indices",      "force_node_indices"]),
            ("contact",  ["contact_node_indices"]),
            ("thermal",  ["thermal_node_indices",   "temperature_node_indices"]),
        ]:
            for key in idx_keys:
                indices = boundary_info.get(key, [])
                if len(indices) > 0:
                    bc_ids[indices] = FEA_BC_TYPES[bc_type]
                    break

        return bc_ids

    def encode_bc_values(self, case: dict, n_nodes: int) -> np.ndarray:
        """
        Encode BC value magnitudes per node as float array.
        Channels: [displacement_mag, force_mag, temperature, contact_pressure]
        """
        bc_vals = np.zeros((n_nodes, 4), dtype=np.float32)
        bcs     = case.get("boundary_conditions", {})
        bi      = case.get("boundary_info", {})

        # Channel 0: prescribed displacement magnitude at fixed/pinned nodes
        fixed_idx = bi.get("fixed_node_indices", [])
        if len(fixed_idx) > 0:
            bc_vals[fixed_idx, 0] = 0.0  # Fixed → zero displacement

        # Channel 1: applied force magnitude
        load_idx = bi.get("load_node_indices", [])
        load_val = bcs.get("load", {}).get("magnitude", 0.0)
        if len(load_idx) > 0:
            bc_vals[load_idx, 1] = float(load_val)

        # Channel 2: prescribed temperature
        temp_idx = bi.get("thermal_node_indices", [])
        temp_val = bcs.get("temperature", {}).get("value", 0.0)
        if len(temp_idx) > 0:
            bc_vals[temp_idx, 2] = float(temp_val)

        return bc_vals
