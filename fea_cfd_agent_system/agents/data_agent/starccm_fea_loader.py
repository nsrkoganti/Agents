"""
STAR-CCM+ FEA Loader — reads STAR-CCM+ structural analysis exports.

STAR-CCM+ has no public Python API. This loader reads CSV or VTK exports
from STAR-CCM+ structural analysis (via File → Export → Field Data or macros).

Detects FEA fields by name patterns from the unified FEA field alias list.
"""

import numpy as np
from loguru import logger


FEA_COLUMN_PATTERNS = {
    "displacement": ["Displacement", "disp", "U[", "ux", "uy", "uz", "deformation"],
    "stress":       ["Stress", "sigma", "S[", "sx", "sy", "sz", "sxx", "syy", "szz"],
    "von_mises":    ["VonMises", "Mises", "eqv", "SEQV", "VM_Stress"],
    "strain":       ["Strain", "epsilon", "E[", "ex", "ey", "ez"],
    "temperature":  ["Temperature", "Temp", "T["],
    "reaction":     ["ReactionForce", "RF[", "reaction"],
    "pressure":     ["ContactPressure", "CPRESS"],
}


class StarCCMFEALoader:

    def load(self, path: str) -> dict:
        if path.endswith(".csv"):
            return self._load_csv(path)
        if path.endswith((".vtk", ".vtu")):
            return self._load_vtk(path)
        # Try CSV first, then VTK
        try:
            return self._load_csv(path)
        except Exception:
            return self._load_vtk(path)

    def _load_csv(self, path: str) -> dict:
        import pandas as pd
        df = pd.read_csv(path)

        # Detect coordinate columns
        coord_cols = self._find_columns(df, ["X", "Y", "Z", "x ", "y ", "z ", "Pos"])
        if not coord_cols or len(coord_cols) < 3:
            logger.warning("STAR-CCM+ CSV: could not detect XYZ coordinate columns")
            coord_cols = df.columns[:3].tolist()

        nodes   = df[coord_cols[:3]].values.astype(np.float32)
        n_nodes = len(nodes)
        fields  = {}

        for std_name, patterns in FEA_COLUMN_PATTERNS.items():
            cols = self._find_columns(df, patterns)
            if cols:
                data = df[cols].values.astype(np.float32)
                fields[std_name] = data if data.shape[1] > 1 else data.ravel()

        return self._build_schema(nodes, fields, n_nodes)

    def _load_vtk(self, path: str) -> dict:
        import pyvista as pv
        mesh    = pv.read(path)
        nodes   = np.array(mesh.points, dtype=np.float32)
        n_nodes = len(nodes)
        fields  = {}

        for std_name, patterns in FEA_COLUMN_PATTERNS.items():
            for store in [mesh.point_data, mesh.cell_data]:
                for key in store.keys():
                    if any(p.lower() in key.lower() for p in patterns):
                        fields[std_name] = np.array(store[key], dtype=np.float32)
                        break
                if std_name in fields:
                    break

        return self._build_schema(nodes, fields, n_nodes)

    def _find_columns(self, df, patterns: list) -> list:
        """Find DataFrame columns matching any pattern (case-insensitive partial match)."""
        found = []
        for col in df.columns:
            col_lower = col.lower()
            if any(p.lower().strip("[]") in col_lower for p in patterns):
                found.append(col)
        return found

    def _build_schema(self, nodes, fields, n_nodes):
        return {
            "nodes":      nodes,
            "elements":   np.array([]),
            "fields":     fields,
            "n_nodes":    n_nodes,
            "n_elements": 0,
            "physics_type":    "FEA_static_linear",
            "solver_source":   "STAR-CCM+",
            "boundary_info":   {},
            "boundary_conditions": {},
            "material_properties": {},
            "mesh_type":       "unstructured_tetrahedral",
        }
