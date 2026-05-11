"""
CalculiX Loader — reads CalculiX FRD result files.

Strategy:
  1. ccx2paraview: converts .frd → .vtu (pip install ccx2paraview)
  2. meshio.read(".frd"): direct FRD reading via meshio
  3. pyvista: reads the converted .vtu

CalculiX field names → unified FEA schema mapping included.
"""

import numpy as np
import os
import tempfile
from loguru import logger


CCX_FIELD_MAP = {
    # CalculiX name → unified name
    "DISP":   "displacement",   # U1, U2, U3
    "STRESS": "stress",         # SXX, SYY, SZZ, SXY, SYZ, SXZ
    "STRAIN": "strain",
    "MISES":  "von_mises",
    "TEMP":   "temperature",
    "RF":     "reaction_forces",
    # Alternative names
    "U":      "displacement",
    "S":      "stress",
    "E":      "strain",
}


class CalculiXLoader:

    def load(self, path: str) -> dict:
        if path.endswith(".frd"):
            return self._load_frd(path)
        if path.endswith(".dat"):
            return self._load_dat(path)
        return None

    def _load_frd(self, path: str) -> dict:
        # Try ccx2paraview conversion first
        vtu_path = self._try_ccx2paraview(path)
        if vtu_path:
            result = self._load_vtu(vtu_path)
            if result:
                return result

        # Direct FRD reading via meshio
        try:
            return self._load_via_meshio(path)
        except Exception as e:
            logger.warning(f"CalculiX FRD meshio load failed: {e}")
            return None

    def _try_ccx2paraview(self, frd_path: str):
        """Convert .frd to .vtu using ccx2paraview if available."""
        try:
            import ccx2paraview
            tmp_dir = tempfile.mkdtemp()
            vtu_path = os.path.join(tmp_dir, "result.vtu")
            ccx2paraview.convert(frd_path, tmp_dir)
            # ccx2paraview writes files with the base name of the input
            base = os.path.splitext(os.path.basename(frd_path))[0]
            vtu_candidate = os.path.join(tmp_dir, f"{base}.vtu")
            if os.path.exists(vtu_candidate):
                return vtu_candidate
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"ccx2paraview conversion failed: {e}")
        return None

    def _load_vtu(self, path: str) -> dict:
        import pyvista as pv
        mesh    = pv.read(path)
        nodes   = np.array(mesh.points)
        n_nodes = len(nodes)
        fields  = {}

        for ccx_name, std_name in CCX_FIELD_MAP.items():
            for data_store in [mesh.point_data, mesh.cell_data]:
                if ccx_name in data_store:
                    fields[std_name] = np.array(data_store[ccx_name])
                    break
                # Also check with component suffixes (U1, U2, U3)
                u1 = f"{ccx_name}1"
                u2 = f"{ccx_name}2"
                u3 = f"{ccx_name}3"
                if u1 in data_store and u2 in data_store and u3 in data_store:
                    fields[std_name] = np.stack([
                        data_store[u1], data_store[u2], data_store[u3]
                    ], axis=-1)
                    break

        return self._build_schema(nodes, fields, n_nodes, mesh.n_cells)

    def _load_via_meshio(self, path: str) -> dict:
        import meshio
        mesh    = meshio.read(path)
        nodes   = np.array(mesh.points)
        n_nodes = len(nodes)
        fields  = {}

        for ccx_name, std_name in CCX_FIELD_MAP.items():
            if hasattr(mesh, 'point_data') and ccx_name in mesh.point_data:
                fields[std_name] = np.array(mesh.point_data[ccx_name])
            elif hasattr(mesh, 'cell_data') and ccx_name in mesh.cell_data:
                data = mesh.cell_data[ccx_name]
                if isinstance(data, list) and len(data) > 0:
                    fields[std_name] = np.concatenate(data)

        return self._build_schema(nodes, fields, n_nodes, 0)

    def _load_dat(self, path: str) -> dict:
        """Load CalculiX .dat file (ASCII tabular output)."""
        fields = {}
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=r'\s+', header=None, comment='*')
            nodes   = df.iloc[:, :3].values if df.shape[1] >= 3 else np.zeros((1, 3))
            n_nodes = len(nodes)
            if df.shape[1] >= 6:
                fields["displacement"] = df.iloc[:, 3:6].values
            return self._build_schema(nodes, fields, n_nodes, 0)
        except Exception as e:
            logger.error(f"CalculiX .dat load failed: {e}")
            return None

    def _build_schema(self, nodes, fields, n_nodes, n_elements):
        return {
            "nodes":      nodes,
            "elements":   np.array([]),
            "fields":     fields,
            "n_nodes":    n_nodes,
            "n_elements": n_elements,
            "physics_type":    "FEA_static_linear",
            "solver_source":   "CalculiX",
            "boundary_info":   {},
            "boundary_conditions": {},
            "material_properties": {},
            "mesh_type":       "unstructured_tetrahedral",
        }
