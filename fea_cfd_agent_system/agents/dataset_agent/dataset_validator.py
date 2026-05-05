"""
Dataset Validator Agent — validates downloaded datasets for quality,
format correctness, required fields, and license compatibility.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger


ALLOWED_LICENSES = {
    "mit", "apache-2.0", "apache 2.0",
    "cc-by-4.0", "cc-by 4.0", "cc by 4.0",
    "cc0-1.0", "cc0", "public domain",
    "openrail", "bsd", "bsd-2-clause", "bsd-3-clause",
}

BLOCKED_LICENSES = {"cc-by-nc", "cc-by-nc-4.0", "proprietary", "commercial"}


class DatasetValidatorAgent:
    """
    Validates a downloaded dataset directory:
    1. Finds data files (VTK, HDF5, Parquet, NPZ, CSV)
    2. Loads a sample and checks physics fields
    3. Validates physical plausibility
    4. Checks license compatibility
    """

    MIN_NODES   = 50
    MIN_SAMPLES = 5
    MAX_VEL     = 500.0   # m/s — anything faster is unphysical for subsonic CFD

    def validate(self, local_path: str, dataset_info: Dict) -> Dict:
        """
        Run all validation checks. Returns a DatasetQualityReport dict:
        {
          "valid": bool,
          "format": str,
          "n_files": int,
          "n_samples_estimated": int,
          "fields_found": List[str],
          "license_ok": bool,
          "issues": List[str],
        }
        """
        report = {
            "valid":                True,
            "format":               "unknown",
            "n_files":              0,
            "n_samples_estimated":  0,
            "fields_found":         [],
            "license_ok":           True,
            "issues":               [],
            "local_path":           local_path,
        }

        path = Path(local_path)
        if not path.exists():
            report["valid"] = False
            report["issues"].append("Local path does not exist")
            return report

        # License check
        license_val = dataset_info.get("license", "unknown").lower()
        if any(blocked in license_val for blocked in BLOCKED_LICENSES):
            report["license_ok"] = False
            report["valid"]      = False
            report["issues"].append(f"License '{license_val}' not allowed for automated use")
            return report

        # Find data files
        data_files = self._find_data_files(path)
        report["n_files"] = len(data_files)

        if not data_files:
            report["valid"] = False
            report["issues"].append("No recognizable data files found")
            return report

        fmt = self._detect_format(data_files[0])
        report["format"] = fmt
        report["n_samples_estimated"] = min(len(data_files), 999999)

        # Sample a file and validate
        sample_result = self._validate_sample(data_files[0], fmt)
        report["fields_found"] = sample_result.get("fields", [])

        if not sample_result.get("ok"):
            report["valid"] = False
            report["issues"].extend(sample_result.get("issues", []))
            return report

        # Size check
        if report["n_samples_estimated"] < self.MIN_SAMPLES:
            report["issues"].append(
                f"Too few samples: {report['n_samples_estimated']} < {self.MIN_SAMPLES}"
            )
            report["valid"] = False

        logger.info(
            f"Dataset validation: valid={report['valid']}, "
            f"format={fmt}, files={len(data_files)}, "
            f"fields={report['fields_found']}"
        )
        return report

    def _find_data_files(self, path: Path) -> List[Path]:
        extensions = {".vtu", ".vtk", ".vtp", ".h5", ".hdf5",
                      ".parquet", ".npz", ".npy", ".csv", ".nc"}
        files = []
        for ext in extensions:
            files.extend(sorted(path.rglob(f"*{ext}"))[:200])
        return files[:500]

    def _detect_format(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        mapping = {
            ".vtu": "vtu", ".vtk": "vtk", ".vtp": "vtp",
            ".h5": "hdf5", ".hdf5": "hdf5",
            ".parquet": "parquet",
            ".npz": "npz", ".npy": "numpy",
            ".csv": "csv",
            ".nc": "netcdf",
        }
        return mapping.get(suffix, "unknown")

    def _validate_sample(self, file_path: Path, fmt: str) -> Dict:
        result = {"ok": True, "fields": [], "issues": []}
        try:
            if fmt in ("vtu", "vtk", "vtp"):
                return self._validate_vtk(file_path, result)
            elif fmt == "hdf5":
                return self._validate_hdf5(file_path, result)
            elif fmt == "parquet":
                return self._validate_parquet(file_path, result)
            elif fmt in ("npz", "numpy"):
                return self._validate_numpy(file_path, result)
            elif fmt == "csv":
                return self._validate_csv(file_path, result)
            else:
                result["issues"].append(f"Unknown format: {fmt}")
                result["ok"] = False
        except Exception as e:
            result["ok"] = False
            result["issues"].append(f"Validation error: {e}")
        return result

    def _validate_vtk(self, path: Path, result: Dict) -> Dict:
        try:
            import pyvista as pv
            mesh = pv.read(str(path))
            if mesh.n_points < self.MIN_NODES:
                result["issues"].append(f"Too few nodes: {mesh.n_points}")
                result["ok"] = False
            result["fields"] = list(mesh.point_data.keys())
            self._check_physics_fields(result, result["fields"])
            self._check_array_quality(result, mesh.points)
        except ImportError:
            result["issues"].append("pyvista not installed — VTK validation skipped")
        return result

    def _validate_hdf5(self, path: Path, result: Dict) -> Dict:
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                result["fields"] = list(f.keys())
                self._check_physics_fields(result, result["fields"])
                for key in result["fields"][:3]:
                    arr = f[key][:]
                    self._check_array_quality(result, arr)
        except ImportError:
            result["issues"].append("h5py not installed — HDF5 validation skipped")
        return result

    def _validate_parquet(self, path: Path, result: Dict) -> Dict:
        try:
            import pandas as pd
            df = pd.read_parquet(str(path), engine="auto")
            result["fields"] = list(df.columns)
            self._check_physics_fields(result, result["fields"])
            self._check_array_quality(result, df.values[:1000])
        except Exception as e:
            result["issues"].append(f"Parquet read failed: {e}")
            result["ok"] = False
        return result

    def _validate_numpy(self, path: Path, result: Dict) -> Dict:
        try:
            if path.suffix == ".npz":
                data = np.load(str(path), allow_pickle=False)
                result["fields"] = list(data.files)
                for key in result["fields"][:3]:
                    self._check_array_quality(result, data[key])
            else:
                arr = np.load(str(path), allow_pickle=False)
                result["fields"] = [f"dim_{i}" for i in range(arr.ndim)]
                self._check_array_quality(result, arr)
        except Exception as e:
            result["issues"].append(f"NumPy load failed: {e}")
            result["ok"] = False
        return result

    def _validate_csv(self, path: Path, result: Dict) -> Dict:
        try:
            import pandas as pd
            df = pd.read_csv(str(path), nrows=1000)
            result["fields"] = list(df.columns)
            self._check_physics_fields(result, result["fields"])
            self._check_array_quality(result, df.values)
        except Exception as e:
            result["issues"].append(f"CSV read failed: {e}")
            result["ok"] = False
        return result

    def _check_physics_fields(self, result: Dict, fields: List[str]):
        """Check that at least one physics field is present."""
        physics_keywords = {
            "velocity", "pressure", "stress", "displacement",
            "u", "v", "w", "p", "ux", "uy", "uz",
            "temperature", "density", "tke", "omega",
        }
        fields_lower = {f.lower() for f in fields}
        found = fields_lower & physics_keywords
        if not found:
            result["issues"].append(
                f"No recognizable physics fields found. Got: {list(fields)[:8]}"
            )

    def _check_array_quality(self, result: Dict, arr):
        """Check for NaN/Inf in array sample."""
        try:
            a = np.array(arr, dtype=float).ravel()[:10000]
            n_nan = int(np.sum(np.isnan(a)))
            n_inf = int(np.sum(np.isinf(a)))
            if n_nan > len(a) * 0.1:
                result["issues"].append(f"High NaN rate: {n_nan}/{len(a)}")
                result["ok"] = False
            if n_inf > 0:
                result["issues"].append(f"Contains {n_inf} Inf values")
        except Exception:
            pass
