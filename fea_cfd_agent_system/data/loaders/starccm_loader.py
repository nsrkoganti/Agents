"""STAR-CCM+ CSV/table export loader."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger


class StarCCMLoader:
    """
    Loads STAR-CCM+ exported CSV tables (Scene→Export→CSV).
    Also handles .csv exports from Reports and Field Functions.
    """

    COORD_COLS  = ["X", "Y", "Z", "X [m]", "Y [m]", "Z [m]",
                   "Centroid[X]", "Centroid[Y]", "Centroid[Z]"]
    VELOCITY_COLS = ["Velocity[i]", "Velocity[j]", "Velocity[k]",
                     "Velocity u", "Velocity v", "Velocity w",
                     "U", "V", "W"]
    PRESSURE_COLS = ["Pressure", "Static Pressure", "p"]
    TEMP_COLS     = ["Temperature", "T", "Static Temperature"]
    TKE_COLS      = ["Turbulent Kinetic Energy", "k", "TKE"]
    OMEGA_COLS    = ["Specific Dissipation Rate", "omega", "Omega"]

    def load(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"STAR-CCM+ file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in (".ccm", ".sim"):
            raise NotImplementedError(
                "Direct .ccm/.sim reading requires STAR-CCM+ API. "
                "Please export to CSV first."
            )
        else:
            return self._load_csv(path)

    def _load_csv(self, path: Path) -> Dict[str, Any]:
        try:
            df = pd.read_csv(path, skipinitialspace=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")

        df.columns = [c.strip() for c in df.columns]

        coords = self._extract_coords(df)
        fields = self._extract_fields(df)

        return {
            "coordinates": coords,
            "fields":      fields,
            "n_nodes":     len(coords),
            "n_cells":     len(coords),
            "file_path":   str(path),
            "format":      "starccm_csv",
            "mesh_type":   "unstructured",
            "available_fields": list(fields.keys()),
            "column_names": list(df.columns),
        }

    def _extract_coords(self, df: pd.DataFrame) -> np.ndarray:
        xyz = []
        for col in self.COORD_COLS:
            if col in df.columns:
                xyz.append(df[col].values)
                if len(xyz) == 3:
                    break

        if len(xyz) == 3:
            return np.stack(xyz, axis=-1).astype(np.float32)
        elif len(xyz) == 2:
            z = np.zeros(len(xyz[0]))
            return np.stack(xyz + [z], axis=-1).astype(np.float32)

        coord_cols = [c for c in df.columns if any(
            x in c.lower() for x in ["x", "y", "z", "centroid", "position"]
        )]
        if len(coord_cols) >= 2:
            arrs = [df[c].values for c in coord_cols[:3]]
            while len(arrs) < 3:
                arrs.append(np.zeros(len(arrs[0])))
            return np.stack(arrs, axis=-1).astype(np.float32)

        n = len(df)
        return np.zeros((n, 3), dtype=np.float32)

    def _extract_fields(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        fields = {}

        vel_cols = [c for c in df.columns if any(
            v in c for v in self.VELOCITY_COLS
        )]
        if len(vel_cols) >= 2:
            vels = [df[c].values for c in vel_cols[:3]]
            while len(vels) < 3:
                vels.append(np.zeros(len(vels[0])))
            fields["velocity"] = np.stack(vels, axis=-1).astype(np.float32)

        for field_key, col_list in [
            ("pressure",    self.PRESSURE_COLS),
            ("temperature", self.TEMP_COLS),
            ("tke",         self.TKE_COLS),
            ("omega",       self.OMEGA_COLS),
        ]:
            for col in col_list:
                if col in df.columns:
                    fields[field_key] = df[col].values.astype(np.float32)
                    break

        skip_prefixes = self.COORD_COLS + self.VELOCITY_COLS
        for col in df.columns:
            if col not in skip_prefixes and col not in fields:
                try:
                    arr = pd.to_numeric(df[col], errors="coerce").values
                    if not np.all(np.isnan(arr)):
                        fields[col.lower().replace(" ", "_")] = arr.astype(np.float32)
                except Exception:
                    pass

        return fields

    def get_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fields_info = {}
        for name, arr in data.get("fields", {}).items():
            fields_info[name] = {
                "shape": list(arr.shape),
                "min":   float(np.nanmin(arr)),
                "max":   float(np.nanmax(arr)),
                "mean":  float(np.nanmean(arr)),
            }
        return {
            "n_nodes":   data["n_nodes"],
            "mesh_type": data["mesh_type"],
            "fields":    fields_info,
            "format":    data["format"],
        }
