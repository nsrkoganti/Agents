"""
Data Agent — ingests FEA simulation data from any supported solver,
validates quality, and outputs a unified schema dataset.
"""

import numpy as np
from pathlib import Path
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus


FORMAT_MAP = {
    ".rst":  "ansys",
    ".rth":  "ansys_thermal",
    ".odb":  "abaqus",
    ".inp":  "abaqus_input",
    ".frd":  "calculix",
    ".dat":  "calculix",
    ".vtu":  "vtk",
    ".vtk":  "vtk",
    ".h5":   "hdf5",
    ".hdf5": "hdf5",
    ".csv":  "csv",
    ".npy":  "numpy",
    ".npz":  "numpy",
}


class DataAgent:
    """
    Reads FEA simulation data from ANSYS, Abaqus, CalculiX, STAR-CCM+, VTK, HDF5.
    Cleans, validates, and produces unified FEA schema dataset.
    """

    def __init__(self, config: dict):
        self.config = config

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info(f"Data Agent: loading from {state.data_path}")
        state.data_agent_status = AgentStatus.RUNNING

        try:
            raw_data = self._load_data(state.data_path, state.software_source)
            accepted, rejected = self._inspect_quality(raw_data)
            logger.info(f"Quality check: {len(accepted)} accepted, {len(rejected)} rejected")

            unified = self._build_unified_schema(accepted, state.software_source)
            state.dataset       = unified
            state.unified_schema = {
                "n_cases":        len(accepted),
                "n_rejected":     len(rejected),
                "software":       state.software_source,
                "fields":         list(unified.get("fields", {}).keys()),
                "n_nodes":        unified.get("n_nodes", 0),
                "n_elements":     unified.get("n_elements", 0),
                "boundary_types": unified.get("boundary_types", []),
                "physics_type":   unified.get("physics_type", ""),
            }
            state.data_agent_status = AgentStatus.PASSED
            logger.success(f"Data Agent complete: {len(accepted)} cases loaded")

        except Exception as e:
            state.data_agent_status = AgentStatus.FAILED
            state.error_message     = f"Data Agent failed: {str(e)}"
            logger.error(state.error_message)

        return state

    def _load_data(self, path: str, software: str) -> list:
        p = Path(path)
        if p.is_dir():
            files = []
            for ext in FORMAT_MAP:
                files.extend(p.glob(f"**/*{ext}"))
        else:
            files = [p]

        results = []
        for f in files:
            if not f.exists():
                continue
            loader = self._get_loader(str(f.suffix).lower(), software)
            try:
                data = loader.load(str(f))
                if isinstance(data, list):
                    results.extend(data)   # NumpyLoader returns a list of case dicts
                elif data:
                    results.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        return results

    def _get_loader(self, ext: str, software: str):
        fmt = FORMAT_MAP.get(ext, "vtk")

        if fmt in ("ansys", "ansys_thermal"):
            from agents.data_agent.ansys_loader import ANSYSLoader
            return ANSYSLoader()
        if fmt in ("abaqus", "abaqus_input"):
            from agents.data_agent.abaqus_loader import AbaqusLoader
            return AbaqusLoader()
        if fmt == "calculix":
            from agents.data_agent.calculix_loader import CalculiXLoader
            return CalculiXLoader()
        if fmt == "csv" and software in ("STAR-CCM+", "STARCCM"):
            from agents.data_agent.starccm_fea_loader import StarCCMFEALoader
            return StarCCMFEALoader()
        if fmt == "numpy":
            from agents.data_agent.numpy_loader import NumpyLoader
            return NumpyLoader()

        # Universal fallback: VTK / HDF5 / CSV
        from data.loaders.vtk_loader import VTKLoader
        return VTKLoader()

    def _inspect_quality(self, cases: list) -> tuple:
        accepted, rejected = [], []
        for case in cases:
            if case is None:
                continue
            issues = []
            for field_name, field_data in case.get("fields", {}).items():
                arr = np.asarray(field_data) if not isinstance(field_data, np.ndarray) else field_data
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    issues.append(f"NaN/Inf in field {field_name}")
            skewness = case.get("mesh_quality", {}).get("skewness_max", 0)
            if skewness > 0.95:
                issues.append(f"Mesh skewness too high: {skewness:.2f}")
            if issues:
                case["rejection_reasons"] = issues
                rejected.append(case)
            else:
                accepted.append(case)
        return accepted, rejected

    def _build_unified_schema(self, cases: list, software: str) -> dict:
        if not cases:
            return {}
        sample = cases[0]
        return {
            "cases":          cases,
            "n_cases":        len(cases),
            "software":       software,
            "fields":         {k: np.asarray(v).shape for k, v in sample.get("fields", {}).items()},
            "n_nodes":        sample.get("n_nodes", 0),
            "n_elements":     sample.get("n_elements", 0),
            "boundary_types": sample.get("boundary_types", []),
            "mesh_type":      sample.get("mesh_type", "unknown"),
            "physics_type":   sample.get("physics_type", "FEA_static_linear"),
            "solver_source":  sample.get("solver_source", software),
        }
