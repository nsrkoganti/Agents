"""ANSYS Fluent/Mechanical result file loader."""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class AnsysLoader:
    """
    Loads ANSYS Fluent (.cas/.dat) and Mechanical (.rst) results.
    Primary path: load VTK export from Fluent.
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ANSYS file not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in (".vtk", ".vtu", ".vtp"):
            from data.loaders.vtk_loader import VTKLoader
            return VTKLoader().load(file_path)

        if suffix == ".csv":
            from data.loaders.starccm_loader import StarCCMLoader
            data = StarCCMLoader().load(file_path)
            data["format"] = "ansys_csv"
            return data

        if suffix in (".cas", ".dat"):
            return self._load_fluent_cas(path)

        if suffix == ".rst":
            return self._load_mechanical_rst(path)

        raise ValueError(f"Unsupported ANSYS format: {suffix}")

    def _load_fluent_cas(self, path: Path) -> Dict[str, Any]:
        logger.warning(
            "Direct .cas/.dat reading is not fully supported. "
            "Export to VTK from Fluent for best results."
        )
        n = 100
        return {
            "coordinates":     np.zeros((n, 3), dtype=np.float32),
            "fields":          {"pressure": np.zeros(n, dtype=np.float32)},
            "n_nodes":         n,
            "n_cells":         n,
            "file_path":       str(path),
            "format":          "ansys_fluent",
            "mesh_type":       "unstructured",
            "available_fields": ["pressure"],
            "warning":         "Placeholder — export to VTK for real data",
        }

    def _load_mechanical_rst(self, path: Path) -> Dict[str, Any]:
        try:
            import ansys.mapdl.reader as pymapdl_reader
            rst = pymapdl_reader.read_binary(str(path))
            coords = np.array(rst.mesh.nodes, dtype=np.float32)
            nnum, stress = rst.nodal_stress(0)
            fields = {"stress": stress.astype(np.float32)}
            return {
                "coordinates":     coords,
                "fields":          fields,
                "n_nodes":         len(coords),
                "n_cells":         len(coords),
                "file_path":       str(path),
                "format":          "ansys_rst",
                "mesh_type":       "unstructured",
                "available_fields": list(fields.keys()),
            }
        except ImportError:
            logger.warning("ansys-mapdl-reader not installed — using placeholder")
            n = 100
            return {
                "coordinates":     np.zeros((n, 3), dtype=np.float32),
                "fields":          {"stress": np.zeros((n, 6), dtype=np.float32)},
                "n_nodes":         n,
                "n_cells":         n,
                "file_path":       str(path),
                "format":          "ansys_rst",
                "mesh_type":       "unstructured",
                "available_fields": ["stress"],
            }
