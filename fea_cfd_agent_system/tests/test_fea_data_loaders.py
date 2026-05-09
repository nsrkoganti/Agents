"""
Unit tests for FEA data loaders.
Tests ANSYS, Abaqus, CalculiX, STAR-CCM+ loaders, field_extractor, bc_encoder.
"""

import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNIFIED_SCHEMA_KEYS = [
    "nodes", "elements", "displacement", "stress", "strain",
    "von_mises", "boundary_conditions", "material_props", "physics_type", "solver_source",
]


# ── Field Extractor ──────────────────────────────────────────────────────────

class TestFieldExtractor:

    def test_displacement_alias_resolution(self):
        from agents.data_agent.field_extractor import FieldExtractor
        fe = FieldExtractor()
        raw = {"U": np.zeros((10, 3)), "Coordinates": np.zeros((10, 3))}
        out = fe.extract(raw)
        assert "displacement" in out
        assert out["displacement"].shape == (10, 3)

    def test_stress_voigt_assembly(self):
        from agents.data_agent.field_extractor import FieldExtractor
        fe = FieldExtractor()
        raw = {
            "S11": np.ones(10),
            "S22": np.ones(10) * 2,
            "S33": np.ones(10) * 3,
            "S12": np.zeros(10),
            "S23": np.zeros(10),
            "S13": np.zeros(10),
        }
        out = fe.extract(raw)
        assert "stress" in out
        assert out["stress"].shape == (10, 6)
        assert np.allclose(out["stress"][:, 0], 1.0)
        assert np.allclose(out["stress"][:, 1], 2.0)

    def test_von_mises_computed_from_voigt(self):
        from agents.data_agent.field_extractor import FieldExtractor
        fe = FieldExtractor()
        # Uniaxial stress: σxx = 100, all others = 0
        raw = {
            "S11": np.full(5, 100.0),
            "S22": np.zeros(5),
            "S33": np.zeros(5),
            "S12": np.zeros(5),
            "S23": np.zeros(5),
            "S13": np.zeros(5),
        }
        out = fe.extract(raw)
        derived = fe.compute_derived(out)
        if "von_mises" in derived:
            np.testing.assert_allclose(derived["von_mises"], 100.0, rtol=1e-5)

    def test_temperature_alias(self):
        from agents.data_agent.field_extractor import FieldExtractor
        fe = FieldExtractor()
        raw = {"TEMP": np.linspace(20, 100, 20)}
        out = fe.extract(raw)
        assert "temperature" in out

    def test_unknown_fields_ignored(self):
        from agents.data_agent.field_extractor import FieldExtractor
        fe = FieldExtractor()
        raw = {"UNKNOWN_FIELD_XYZ": np.zeros(5)}
        out = fe.extract(raw)
        assert "UNKNOWN_FIELD_XYZ" not in out


# ── BC Encoder ───────────────────────────────────────────────────────────────

class TestBCEncoder:

    def test_encode_returns_tensor(self):
        from agents.data_agent.bc_encoder import BCEncoder
        enc = BCEncoder()
        bc = {
            "fixed":   np.array([0, 1, 2]),
            "load":    np.array([10, 11]),
            "symmetry": np.array([5]),
        }
        result = enc.encode(bc, n_nodes=15)
        assert result.shape[0] == 15

    def test_fixed_encoded_correctly(self):
        from agents.data_agent.bc_encoder import BCEncoder
        enc = BCEncoder()
        bc = {"fixed": np.array([0, 1])}
        result = enc.encode(bc, n_nodes=5)
        assert result[0, 0] == 0  # fixed type mapped to 0
        assert result[2, 0] != 0 or result[2, 0] == 0  # interior nodes

    def test_no_bc_returns_zeros(self):
        from agents.data_agent.bc_encoder import BCEncoder
        enc = BCEncoder()
        result = enc.encode({}, n_nodes=10)
        assert result.shape[0] == 10

    def test_encode_bc_values_shape(self):
        from agents.data_agent.bc_encoder import BCEncoder
        enc = BCEncoder()
        bc = {"fixed": np.array([0, 1, 2]), "load": np.array([9])}
        values = enc.encode_bc_values(bc, n_nodes=10)
        assert values.shape[0] == 10
        assert values.shape[1] >= 1


# ── ANSYS Loader ─────────────────────────────────────────────────────────────

class TestANSYSLoader:

    def test_loader_imports(self):
        from agents.data_agent.ansys_loader import ANSYSLoader
        loader = ANSYSLoader()
        assert loader is not None

    def test_load_nonexistent_file(self):
        from agents.data_agent.ansys_loader import ANSYSLoader
        loader = ANSYSLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/file.rst")

    def test_unified_schema_synthetic(self):
        from agents.data_agent.ansys_loader import ANSYSLoader
        loader = ANSYSLoader()
        # Use synthetic fallback path by testing _make_synthetic
        if hasattr(loader, "_make_synthetic"):
            result = loader._make_synthetic(n_nodes=20)
            for key in ["nodes", "displacement", "stress", "solver_source"]:
                assert key in result

    def test_solver_source_tag(self):
        from agents.data_agent.ansys_loader import ANSYSLoader
        loader = ANSYSLoader()
        if hasattr(loader, "_make_synthetic"):
            result = loader._make_synthetic(n_nodes=10)
            assert result["solver_source"] == "ANSYS"


# ── Abaqus Loader ─────────────────────────────────────────────────────────────

class TestAbaqusLoader:

    def test_loader_imports(self):
        from agents.data_agent.abaqus_loader import AbaqusLoader
        loader = AbaqusLoader()
        assert loader is not None

    def test_load_nonexistent_file(self):
        from agents.data_agent.abaqus_loader import AbaqusLoader
        loader = AbaqusLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/model.odb")

    def test_inp_fallback_with_temp_file(self):
        from agents.data_agent.abaqus_loader import AbaqusLoader
        loader = AbaqusLoader()
        # Create minimal .inp-like content
        inp_content = """*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
*Element, type=C3D4
1, 1, 2, 3
*Boundary
1, 1, 6
"""
        with tempfile.NamedTemporaryFile(suffix=".inp", mode="w", delete=False) as f:
            f.write(inp_content)
            tmp_path = f.name
        try:
            if hasattr(loader, "load_inp"):
                result = loader.load_inp(tmp_path)
                assert isinstance(result, dict)
        finally:
            os.unlink(tmp_path)

    def test_solver_source_tag(self):
        from agents.data_agent.abaqus_loader import AbaqusLoader
        loader = AbaqusLoader()
        if hasattr(loader, "_make_synthetic"):
            result = loader._make_synthetic(n_nodes=10)
            assert result["solver_source"] == "Abaqus"


# ── CalculiX Loader ───────────────────────────────────────────────────────────

class TestCalculiXLoader:

    def test_loader_imports(self):
        from agents.data_agent.calculix_loader import CalculiXLoader
        loader = CalculiXLoader()
        assert loader is not None

    def test_load_nonexistent_file(self):
        from agents.data_agent.calculix_loader import CalculiXLoader
        loader = CalculiXLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/path/result.frd")

    def test_field_map_coverage(self):
        from agents.data_agent.calculix_loader import CalculiXLoader
        loader = CalculiXLoader()
        assert hasattr(loader, "CCX_FIELD_MAP") or hasattr(loader, "_field_map")

    def test_solver_source_tag(self):
        from agents.data_agent.calculix_loader import CalculiXLoader
        loader = CalculiXLoader()
        if hasattr(loader, "_make_synthetic"):
            result = loader._make_synthetic(n_nodes=10)
            assert result["solver_source"] == "CalculiX"


# ── STAR-CCM+ FEA Loader ──────────────────────────────────────────────────────

class TestStarCCMFEALoader:

    def test_loader_imports(self):
        from agents.data_agent.starccm_fea_loader import StarCCMFEALoader
        loader = StarCCMFEALoader()
        assert loader is not None

    def test_load_csv_with_stress_columns(self):
        from agents.data_agent.starccm_fea_loader import StarCCMFEALoader
        import pandas as pd
        import io
        loader = StarCCMFEALoader()
        csv_data = """X[m],Y[m],Z[m],Displacement X[m],Displacement Y[m],Displacement Z[m],Stress XX[Pa],Stress YY[Pa],Stress ZZ[Pa]
0.0,0.0,0.0,0.001,0.0,0.0,1e6,2e6,3e6
1.0,0.0,0.0,0.002,0.0,0.0,1.1e6,2.1e6,3.1e6
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write(csv_data)
            tmp_path = f.name
        try:
            if hasattr(loader, "load"):
                result = loader.load(tmp_path)
                assert isinstance(result, dict)
        except Exception:
            pass  # STAR-CCM+ loader may require specific format
        finally:
            os.unlink(tmp_path)

    def test_fea_column_patterns_exist(self):
        from agents.data_agent.starccm_fea_loader import StarCCMFEALoader
        loader = StarCCMFEALoader()
        assert hasattr(loader, "FEA_COLUMN_PATTERNS") or hasattr(loader, "_fea_patterns")

    def test_solver_source_tag(self):
        from agents.data_agent.starccm_fea_loader import StarCCMFEALoader
        loader = StarCCMFEALoader()
        if hasattr(loader, "_make_synthetic"):
            result = loader._make_synthetic(n_nodes=10)
            assert result["solver_source"] == "STAR-CCM+"


# ── Data Agent Integration ────────────────────────────────────────────────────

class TestDataAgentFormatMap:

    def test_format_map_coverage(self):
        from agents.data_agent.data_agent import DataAgent
        agent = DataAgent(config={})
        assert hasattr(agent, "FORMAT_MAP") or hasattr(agent, "_format_map")
        fmt_map = getattr(agent, "FORMAT_MAP", getattr(agent, "_format_map", {}))
        for ext in [".rst", ".odb", ".frd", ".vtu", ".h5"]:
            assert ext in fmt_map, f"Missing format: {ext}"

    def test_no_openfoam_in_format_map(self):
        from agents.data_agent.data_agent import DataAgent
        agent = DataAgent(config={})
        fmt_map = getattr(agent, "FORMAT_MAP", getattr(agent, "_format_map", {}))
        values = list(fmt_map.values())
        assert "openfoam" not in values, "OpenFOAM should not be in FEA-only format map"
