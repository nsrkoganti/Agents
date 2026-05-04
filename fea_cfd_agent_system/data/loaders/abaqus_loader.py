"""Abaqus ODB result loader."""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class AbaqusLoader:
    """
    Loads Abaqus .odb files or CSV exports from Abaqus/CAE.
    Direct .odb reading requires Abaqus Python API; CSV export is the safe path.
    """

    def load(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Abaqus file not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".vtk", ".vtu"):
            from data.loaders.vtk_loader import VTKLoader
            return VTKLoader().load(file_path)
        elif suffix == ".odb":
            return self._load_odb(path)
        else:
            raise ValueError(f"Unsupported Abaqus format: {suffix}")

    def _load_csv(self, path: Path) -> Dict[str, Any]:
        import pandas as pd
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        coord_cols = [c for c in df.columns if c.upper() in ["X", "Y", "Z"]]
        if len(coord_cols) >= 2:
            arrs = [df[c].values for c in coord_cols[:3]]
            while len(arrs) < 3:
                arrs.append(np.zeros(len(arrs[0])))
            coords = np.stack(arrs, axis=-1).astype(np.float32)
        else:
            n = len(df)
            coords = np.zeros((n, 3), dtype=np.float32)

        fields = {}
        skip = set(coord_cols + ["Node Label", "Element Label"])
        for col in df.columns:
            if col not in skip:
                try:
                    arr = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
                    fields[col.lower().replace(" ", "_")] = arr
                except Exception:
                    pass

        return {
            "coordinates":     coords,
            "fields":          fields,
            "n_nodes":         len(coords),
            "n_cells":         len(coords),
            "file_path":       str(path),
            "format":          "abaqus_csv",
            "mesh_type":       "unstructured",
            "available_fields": list(fields.keys()),
        }

    def _load_odb(self, path: Path) -> Dict[str, Any]:
        logger.warning(
            "Direct .odb reading requires Abaqus Python API. "
            "Export results to CSV or VTK from Abaqus/CAE."
        )
        n = 100
        return {
            "coordinates":     np.zeros((n, 3), dtype=np.float32),
            "fields": {
                "s_mises": np.zeros(n, dtype=np.float32),
                "u_magnitude": np.zeros(n, dtype=np.float32),
            },
            "n_nodes":         n,
            "n_cells":         n,
            "file_path":       str(path),
            "format":          "abaqus_odb",
            "mesh_type":       "unstructured",
            "available_fields": ["s_mises", "u_magnitude"],
            "warning":         "Placeholder — export to CSV or VTK for real data",
        }
