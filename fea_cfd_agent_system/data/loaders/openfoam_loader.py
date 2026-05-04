"""OpenFOAM case directory loader."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class OpenFOAMLoader:
    """
    Loads OpenFOAM case directories.
    Reads the latest time directory for field data.
    Falls back to VTK export if available.
    """

    VECTOR_FIELDS  = {"U", "velocity"}
    SCALAR_FIELDS  = {"p", "k", "epsilon", "omega", "T",
                      "nut", "nuTilda", "alphat"}

    def load(self, case_dir: str) -> Dict[str, Any]:
        path = Path(case_dir)
        if not path.exists():
            raise FileNotFoundError(f"OpenFOAM case not found: {case_dir}")

        vtk_dir = path / "VTK"
        if vtk_dir.exists():
            vtk_files = list(vtk_dir.glob("*.vtk")) + list(vtk_dir.glob("*.vtu"))
            if vtk_files:
                logger.info(f"Loading from VTK export: {vtk_files[-1]}")
                from data.loaders.vtk_loader import VTKLoader
                return VTKLoader().load(str(vtk_files[-1]))

        time_dir = self._find_latest_time(path)
        if time_dir is None:
            raise RuntimeError(f"No time directories found in {case_dir}")

        logger.info(f"Reading OpenFOAM time directory: {time_dir}")
        points, n_points = self._read_points(path)
        fields = self._read_all_fields(time_dir)

        return {
            "coordinates": points,
            "fields":      fields,
            "n_nodes":     n_points,
            "n_cells":     n_points,
            "file_path":   str(path),
            "format":      "openfoam",
            "mesh_type":   "unstructured",
            "available_fields": list(fields.keys()),
        }

    def _find_latest_time(self, case_dir: Path) -> Optional[Path]:
        time_dirs = []
        for d in case_dir.iterdir():
            if d.is_dir():
                try:
                    t = float(d.name)
                    time_dirs.append((t, d))
                except ValueError:
                    pass
        if not time_dirs:
            return None
        time_dirs.sort(key=lambda x: x[0])
        return time_dirs[-1][1]

    def _read_points(self, case_dir: Path):
        points_file = case_dir / "constant" / "polyMesh" / "points"
        if not points_file.exists():
            points_file = case_dir / "constant" / "polyMesh" / "points.gz"

        if not points_file.exists():
            logger.warning("No OpenFOAM points file found — returning placeholder")
            return np.zeros((1, 3), dtype=np.float32), 1

        try:
            content = self._read_foam_file(points_file)
            pts = self._parse_vector_list(content)
            return np.array(pts, dtype=np.float32), len(pts)
        except Exception as e:
            logger.error(f"Failed to read points: {e}")
            return np.zeros((1, 3), dtype=np.float32), 1

    def _read_all_fields(self, time_dir: Path) -> Dict[str, np.ndarray]:
        fields = {}
        for field_file in time_dir.iterdir():
            if field_file.is_file() and not field_file.name.startswith("."):
                try:
                    data = self._read_field_file(field_file)
                    if data is not None:
                        fields[field_file.name.lower()] = data
                except Exception as e:
                    logger.debug(f"Could not read field {field_file.name}: {e}")
        return fields

    def _read_field_file(self, path: Path) -> Optional[np.ndarray]:
        content = self._read_foam_file(path)
        if "internalField" not in content:
            return None

        internal_section = content.split("internalField")[1]

        if "uniform" in internal_section[:50]:
            return None

        if "nonuniform" in internal_section[:50]:
            if "(" in internal_section:
                first_paren = internal_section.find("(")
                list_section = internal_section[first_paren:]
                if list_section.startswith("(("):
                    return np.array(
                        self._parse_vector_list(list_section), dtype=np.float32
                    )
                else:
                    return np.array(
                        self._parse_scalar_list(list_section), dtype=np.float32
                    )
        return None

    def _read_foam_file(self, path: Path) -> str:
        try:
            if str(path).endswith(".gz"):
                import gzip
                with gzip.open(path, "rt") as f:
                    return f.read()
            else:
                with open(path, "r", errors="ignore") as f:
                    return f.read()
        except Exception:
            return ""

    def _parse_vector_list(self, content: str) -> List[List[float]]:
        import re
        pattern = r'\(([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\)'
        matches = re.findall(pattern, content)
        return [[float(x), float(y), float(z)] for x, y, z in matches]

    def _parse_scalar_list(self, content: str) -> List[float]:
        import re
        numbers = re.findall(r'[+-]?[\d.]+(?:[eE][+-]?\d+)?', content)
        return [float(n) for n in numbers if n]

    def get_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fields_info = {}
        for name, arr in data.get("fields", {}).items():
            fields_info[name] = {
                "shape": list(arr.shape),
                "min":   float(np.nanmin(arr)),
                "max":   float(np.nanmax(arr)),
            }
        return {
            "n_nodes":   data["n_nodes"],
            "mesh_type": data["mesh_type"],
            "fields":    fields_info,
            "format":    data["format"],
        }
