"""
Data Agent — ingests simulation data from any software,
validates quality, and outputs a unified schema dataset.
"""

import os
import numpy as np
from pathlib import Path
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus
from data.loaders.starccm_loader import StarCCMLoader
from data.loaders.vtk_loader import VTKLoader
from data.loaders.openfoam_loader import OpenFOAMLoader


class DataAgent:
    """
    Reads simulation data from any supported software.
    Cleans, validates, and produces unified schema dataset.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv":  "starccm",
        ".vtk":  "vtk",
        ".vtu":  "vtk",
        ".cgns": "vtk",
        ".h5":   "hdf5",
    }

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
            state.dataset = unified
            state.unified_schema = {
                "n_cases": len(accepted),
                "n_rejected": len(rejected),
                "software": state.software_source,
                "fields": list(unified.get("fields", {}).keys()),
                "n_nodes": unified.get("n_nodes", 0),
                "n_cells": unified.get("n_cells", 0),
                "boundary_types": unified.get("boundary_types", []),
            }
            state.data_agent_status = AgentStatus.PASSED
            logger.success(f"Data Agent complete: {len(accepted)} cases loaded")

        except Exception as e:
            state.data_agent_status = AgentStatus.FAILED
            state.error_message = f"Data Agent failed: {str(e)}"
            logger.error(state.error_message)

        return state

    def _load_data(self, path: str, software: str) -> list:
        """Load all simulation files from the given path."""
        p = Path(path)
        if p.is_dir():
            files = list(p.glob("**/*.vtk")) + list(p.glob("**/*.vtu")) + list(p.glob("**/*.csv"))
        else:
            files = [p]

        loader_map = {
            "STAR-CCM+": StarCCMLoader,
            "OpenFOAM":  OpenFOAMLoader,
            "default":   VTKLoader,
        }
        loader_cls = loader_map.get(software, loader_map["default"])
        loader = loader_cls()
        return [loader.load(str(f)) for f in files if f.exists()]

    def _inspect_quality(self, cases: list) -> tuple:
        """Filter out bad simulation cases."""
        accepted, rejected = [], []
        for case in cases:
            if case is None:
                continue
            issues = []
            for field_name, field_data in case.get("fields", {}).items():
                if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
                    issues.append(f"NaN/Inf in field {field_name}")
            skewness = case.get("mesh_quality", {}).get("skewness_max", 0)
            if skewness > 0.95:
                issues.append(f"Skewness too high: {skewness:.2f}")
            if issues:
                case["rejection_reasons"] = issues
                rejected.append(case)
            else:
                accepted.append(case)
        return accepted, rejected

    def _build_unified_schema(self, cases: list, software: str) -> dict:
        """Build unified dataset schema from accepted cases."""
        if not cases:
            return {}
        sample = cases[0]
        return {
            "cases": cases,
            "n_cases": len(cases),
            "software": software,
            "fields": {k: v.shape for k, v in sample.get("fields", {}).items()},
            "n_nodes": sample.get("n_nodes", 0),
            "n_cells": sample.get("n_cells", 0),
            "boundary_types": sample.get("boundary_types", []),
            "mesh_type": sample.get("mesh_type", "unknown"),
        }
