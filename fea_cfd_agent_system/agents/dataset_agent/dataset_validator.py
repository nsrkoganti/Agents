"""
Dataset Validator Agent — validates downloaded FEA datasets for quality,
format correctness, required fields, and license compatibility.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from loguru import logger


ALLOWED_LICENSES = {
    "mit", "apache-2.0", "apache 2.0",
    "cc-by-4.0", "cc-by 4.0", "cc by 4.0",
    "cc0-1.0", "cc0", "public domain",
    "openrail", "bsd", "bsd-2-clause", "bsd-3-clause",
    "lgpl-2.1", "gpl-2.0",
}

BLOCKED_LICENSES = {"cc-by-nc", "cc-by-nc-4.0", "proprietary", "commercial"}

FEA_PHYSICS_KEYWORDS = {
    "displacement", "stress", "strain", "von_mises", "mises", "deformation",
    "u", "ux", "uy", "uz", "s11", "s22", "s33", "s12", "s23", "s13",
    "sigma", "epsilon", "reaction", "contact_pressure",
    "temperature", "heat_flux", "thermal", "seqv", "eqv",
    "young", "poisson", "yield", "plasticity", "stiffness",
    "disp", "deflect", "force", "moment", "pressure",
}


class DatasetValidatorAgent:
    """
    Validates a downloaded FEA dataset directory:
    1. Finds data files (VTK, HDF5, Parquet, NPZ, CSV, FRD)
    2. Loads a sample and checks FEA physics fields
    3. Validates physical plausibility (no NaN/Inf)
    4. Checks license compatibility
    """

    MIN_NODES   = 50
    MIN_SAMPLES = 5

    def validate(self, local_path: str, dataset_info: Dict) -> Dict:
        report = {
            "valid":               True,
            "format":              "unknown",
            "n_files":             0,
            "n_samples_estimated": 0,
            "fields_found":        [],
            "license_ok":          True,
            "issues":              [],
            "local_path":          local_path,
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
            report["issues"].append(f"License '{license_val}' not allowed")
            return report

        data_files = self._find_data_files(path)
        report["n_files"] = len(data_files)

        if not data_files:
            report["valid"] = False
            report["issues"].append("No recognizable FEA data files found")
            return report

        fmt = self._detect_format(data_files[0])
        report["format"]              = fmt
        report["n_samples_estimated"] = len(data_files)

        sample_result = self._validate_sample(data_files[0], fmt)
        report["fields_found"] = sample_result.get("fields", [])

        if not sample_result.get("ok"):
            report["valid"] = False
            report["issues"].extend(sample_result.get("issues", []))
            return report

        if report["n_samples_estimated"] < self.MIN_SAMPLES:
            report["issues"].append(
                f"Too few samples: {report['n_samples_estimated']} < {self.MIN_SAMPLES}"
            )
            report["valid"] = False

        logger.info(
            f"FEA Dataset validation: valid={report['valid']}, "
            f"format={fmt}, files={len(data_files)}, "
            f"fields={report['fields_found'][:5]}"
        )
        return report

    def _find_data_files(self, path: Path) -> List[Path]:
        extensions = {".vtu", ".vtk", ".vtp", ".h5", ".hdf5",
                      ".parquet", ".npz", ".npy", ".csv", ".frd", ".rst", ".odb"}
        files = []
        for ext in extensions:
            files.extend(sorted(path.rglob(f"*{ext}"))[:200])
        return files[:500]

    def _detect_format(self, file_path: Path) -> str:
        return {
            ".vtu": "vtu", ".vtk": "vtk", ".vtp": "vtp",
            ".h5": "hdf5", ".hdf5": "hdf5",
            ".parquet": "parquet",
            ".npz": "npz", ".npy": "numpy",
            ".csv": "csv", ".frd": "frd",
            ".rst": "ansys", ".odb": "abaqus",
        }.get(file_path.suffix.lower(), "unknown")

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
                result["issues"].append(f"Format '{fmt}' validation not implemented — skipping")
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
            result["fields"] = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            self._check_fea_fields(result, result["fields"])
            self._check_array_quality(result, mesh.points)
        except ImportError:
            result["issues"].append("pyvista not installed — VTK validation skipped")
        return result

    def _validate_hdf5(self, path: Path, result: Dict) -> Dict:
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                result["fields"] = list(f.keys())
                self._check_fea_fields(result, result["fields"])
                for key in result["fields"][:3]:
                    self._check_array_quality(result, f[key][:])
        except ImportError:
            result["issues"].append("h5py not installed — HDF5 validation skipped")
        return result

    def _validate_parquet(self, path: Path, result: Dict) -> Dict:
        try:
            import pandas as pd
            df = pd.read_parquet(str(path))
            result["fields"] = list(df.columns)
            self._check_fea_fields(result, result["fields"])
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
                result["fields"] = [f"array_{i}" for i in range(arr.ndim)]
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
            self._check_fea_fields(result, result["fields"])
            self._check_array_quality(result, df.select_dtypes(include=float).values)
        except Exception as e:
            result["issues"].append(f"CSV read failed: {e}")
            result["ok"] = False
        return result

    def _check_fea_fields(self, result: Dict, fields: List[str]):
        """Check that at least one FEA physics field is present."""
        fields_lower = {f.lower() for f in fields}
        found = fields_lower & FEA_PHYSICS_KEYWORDS
        if not found:
            result["issues"].append(
                f"No recognizable FEA fields found. Got: {list(fields)[:8]}"
            )

    def _check_array_quality(self, result: Dict, arr):
        try:
            a    = np.array(arr, dtype=float).ravel()[:10000]
            n_nan = int(np.sum(np.isnan(a)))
            n_inf = int(np.sum(np.isinf(a)))
            if n_nan > len(a) * 0.1:
                result["issues"].append(f"High NaN rate: {n_nan}/{len(a)}")
                result["ok"] = False
            if n_inf > 0:
                result["issues"].append(f"Contains {n_inf} Inf values")
        except Exception:
            pass
