"""VTK/VTU file loader — primary format for OpenFOAM, Fluent, STAR-CCM+."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class VTKLoader:
    """
    Loads VTK/VTU simulation output files into unified numpy arrays.
    Supports: .vtk, .vtu, .vtp formats.
    """

    FIELD_ALIASES = {
        "velocity":    ["U", "Velocity", "velocity", "vel", "VELOCITY"],
        "pressure":    ["p", "P", "Pressure", "pressure", "PRESSURE"],
        "temperature": ["T", "Temperature", "temperature", "TEMPERATURE"],
        "tke":         ["k", "TKE", "tke", "turbulent_kinetic_energy"],
        "omega":       ["omega", "Omega", "specific_dissipation_rate"],
        "epsilon":     ["epsilon", "Epsilon", "dissipation_rate"],
        "stress":      ["S", "stress", "Stress", "vonMises"],
        "displacement":["U", "displacement", "disp", "Displacement"],
    }

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a VTK file and return unified data dict.
        Falls back to synthetic data if pyvista not available.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"VTK file not found: {file_path}")

        try:
            import pyvista as pv
            mesh = pv.read(str(path))
            return self._extract_from_pyvista(mesh, path)
        except ImportError:
            logger.warning("pyvista not available — using meshio fallback")
            return self._load_via_meshio(path)
        except Exception as e:
            logger.error(f"VTK load failed: {e}")
            raise

    def _extract_from_pyvista(self, mesh, path: Path) -> Dict[str, Any]:
        coords = np.array(mesh.points, dtype=np.float32)
        fields = {}

        for field_key, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias in mesh.point_data:
                    arr = np.array(mesh.point_data[alias], dtype=np.float32)
                    fields[field_key] = arr
                    break
            if field_key not in fields:
                for alias in aliases:
                    if alias in mesh.cell_data:
                        arr = np.array(mesh.cell_data[alias], dtype=np.float32)
                        fields[field_key] = arr
                        break

        n_cells  = mesh.n_cells
        n_points = mesh.n_points

        return {
            "coordinates": coords,
            "fields":      fields,
            "n_nodes":     n_points,
            "n_cells":     n_cells,
            "bounds":      list(mesh.bounds),
            "file_path":   str(path),
            "format":      "vtk",
            "mesh_type":   self._infer_mesh_type(mesh),
            "available_fields": list(fields.keys()),
        }

    def _load_via_meshio(self, path: Path) -> Dict[str, Any]:
        try:
            import meshio
            mesh = meshio.read(str(path))
            coords = np.array(mesh.points, dtype=np.float32)
            fields = {}
            for key, arr in mesh.point_data.items():
                fields[key.lower()] = np.array(arr, dtype=np.float32)
            return {
                "coordinates": coords,
                "fields":      fields,
                "n_nodes":     len(coords),
                "n_cells":     sum(len(b.data) for b in mesh.cells),
                "file_path":   str(path),
                "format":      "vtk_meshio",
                "mesh_type":   "unstructured",
                "available_fields": list(fields.keys()),
            }
        except Exception as e:
            raise RuntimeError(f"meshio fallback also failed: {e}")

    def _infer_mesh_type(self, mesh) -> str:
        cell_types = set(mesh.celltypes) if hasattr(mesh, "celltypes") else set()
        structured_types = {8, 9, 11, 12}  # VTK hex/quad cell type IDs
        if cell_types and cell_types.issubset(structured_types):
            return "structured"
        return "unstructured"

    def get_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build schema summary for the analyst agent."""
        fields_info = {}
        for name, arr in data.get("fields", {}).items():
            fields_info[name] = {
                "shape": list(arr.shape),
                "min":   float(np.nanmin(arr)),
                "max":   float(np.nanmax(arr)),
                "mean":  float(np.nanmean(arr)),
                "has_nan": bool(np.any(np.isnan(arr))),
            }
        return {
            "n_nodes":   data.get("n_nodes", 0),
            "n_cells":   data.get("n_cells", 0),
            "mesh_type": data.get("mesh_type", "unknown"),
            "fields":    fields_info,
            "bounds":    data.get("bounds", []),
            "format":    data.get("format", "vtk"),
        }
