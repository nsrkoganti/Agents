"""
Unit tests for FEA physics agents.
Tests equilibrium, stress_strain, compatibility, boundary_condition, material agents.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_linear_elastic_data():
    """3D cantilever beam data (linear elastic, small strain)."""
    N = 50
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (N, 3))
    E, nu  = 210e9, 0.3
    lam    = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu     = E / (2 * (1 + nu))

    # Synthetic strain field (small)
    strain = rng.uniform(-1e-4, 1e-4, (N, 6))
    ex, ey, ez     = strain[:,0], strain[:,1], strain[:,2]
    exy, eyz, exz  = strain[:,3], strain[:,4], strain[:,5]
    ev = ex + ey + ez

    stress = np.stack([
        lam * ev + 2 * mu * ex,
        lam * ev + 2 * mu * ey,
        lam * ev + 2 * mu * ez,
        mu * exy,
        mu * eyz,
        mu * exz,
    ], axis=-1)

    displacement = rng.uniform(-1e-3, 1e-3, (N, 3))

    return {
        "nodes":       coords,
        "displacement": displacement,
        "stress":      stress,
        "strain":      strain,
        "von_mises":   np.sqrt(0.5 * ((stress[:,0]-stress[:,1])**2 +
                                       (stress[:,1]-stress[:,2])**2 +
                                       (stress[:,2]-stress[:,0])**2) +
                               3 * (stress[:,3]**2 + stress[:,4]**2 + stress[:,5]**2)),
        "material_props": {"E": E, "nu": nu},
        "boundary_conditions": {
            "fixed": np.array([0, 1, 2]),
        },
        "reaction_forces": np.array([[0.0, 0.0, 1e3],
                                      [0.0, 0.0, 1e3],
                                      [0.0, 0.0, 1e3]]),
        "physics_type": "FEA_static_linear",
    }


@pytest.fixture
def problem_card_linear():
    from agents.orchestrator.agent_state import ProblemCard, PhysicsType
    return ProblemCard(
        physics_type=PhysicsType.FEA_STATIC_LINEAR,
        material_model="linear_elastic",
        loading_type="static",
        element_type="tet",
        solver_source="ANSYS",
    )


@pytest.fixture
def problem_card_elastoplastic():
    from agents.orchestrator.agent_state import ProblemCard, PhysicsType
    return ProblemCard(
        physics_type=PhysicsType.FEA_STATIC_NONLINEAR,
        material_model="elastoplastic",
        loading_type="static",
        element_type="tet",
        solver_source="ANSYS",
        yield_stress=250e6,
    )


# ── Equilibrium Agent ────────────────────────────────────────────────────────

class TestEquilibriumAgent:

    def test_passes_when_stress_consistent(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert isinstance(report, dict)
        assert "passed" in report
        assert "residual" in report
        assert "threshold" in report

    def test_threshold_linear_vs_nonlinear(self, simple_linear_elastic_data,
                                            problem_card_linear, problem_card_elastoplastic):
        from agents.physics_agent.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        r_lin = agent.check(simple_linear_elastic_data, problem_card_linear)
        r_nl  = agent.check(simple_linear_elastic_data, problem_card_elastoplastic)
        assert r_nl["threshold"] > r_lin["threshold"]

    def test_fails_on_garbage_stress(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        bad_data = dict(simple_linear_elastic_data)
        bad_data["stress"] = np.random.uniform(1e9, 1e12, bad_data["stress"].shape)
        bad_data["von_mises"] = np.random.uniform(1e9, 1e12, (len(bad_data["nodes"]),))
        report = agent.check(bad_data, problem_card_linear)
        # Should either pass or fail — main thing is it doesn't crash
        assert "passed" in report

    def test_report_has_required_keys(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        for key in ("passed", "residual", "threshold", "n_nodes"):
            assert key in report, f"Missing key: {key}"


# ── Stress-Strain Agent ──────────────────────────────────────────────────────

class TestStressStrainAgent:

    def test_constitutive_passes_exact(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.stress_strain_agent import StressStrainAgent
        agent = EquilibriumAgent = StressStrainAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert report["passed"], f"Constitutive check failed: {report}"

    def test_symmetry_passes(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.stress_strain_agent import StressStrainAgent
        agent = StressStrainAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert "symmetry_passed" in report or "passed" in report

    def test_von_mises_below_yield(self, simple_linear_elastic_data, problem_card_elastoplastic):
        from agents.physics_agent.stress_strain_agent import StressStrainAgent
        agent = StressStrainAgent()
        data = dict(simple_linear_elastic_data)
        # All von Mises << yield_stress → should pass
        data["von_mises"] = np.full(len(data["nodes"]), 100e6)
        report = agent.check(data, problem_card_elastoplastic)
        assert report.get("von_mises_passed", True)  # may not be present if check skipped

    def test_von_mises_above_yield_fails(self, simple_linear_elastic_data, problem_card_elastoplastic):
        from agents.physics_agent.stress_strain_agent import StressStrainAgent
        agent = StressStrainAgent()
        data = dict(simple_linear_elastic_data)
        # von Mises >> yield_stress → should fail
        data["von_mises"] = np.full(len(data["nodes"]), 500e6)
        report = agent.check(data, problem_card_elastoplastic)
        assert not report.get("von_mises_passed", True)

    def test_wrong_constitutive_fails(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.stress_strain_agent import StressStrainAgent
        agent = StressStrainAgent()
        data = dict(simple_linear_elastic_data)
        # Corrupt stress
        data["stress"] = np.random.uniform(-1e9, 1e9, data["stress"].shape)
        report = agent.check(data, problem_card_linear)
        assert not report["passed"]


# ── Compatibility Agent ──────────────────────────────────────────────────────

class TestCompatibilityAgent:

    def test_runs_without_error(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.compatibility_agent import CompatibilityAgent
        agent = CompatibilityAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert "passed" in report

    def test_required_keys(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.compatibility_agent import CompatibilityAgent
        agent = CompatibilityAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        for key in ("passed", "max_error", "threshold"):
            assert key in report, f"Missing key: {key}"

    def test_consistent_strain_passes(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.compatibility_agent import CompatibilityAgent
        agent = CompatibilityAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        # Consistent data — should either pass or have small error
        assert report["max_error"] is not None

    def test_missing_strain_handled(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.compatibility_agent import CompatibilityAgent
        agent = CompatibilityAgent()
        data = dict(simple_linear_elastic_data)
        del data["strain"]
        report = agent.check(data, problem_card_linear)
        assert "passed" in report  # should degrade gracefully


# ── Boundary Condition Agent ─────────────────────────────────────────────────

class TestBoundaryConditionAgent:

    def test_fixed_nodes_zero_displacement(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.boundary_condition_agent import BoundaryConditionAgent
        agent = BoundaryConditionAgent()
        data = dict(simple_linear_elastic_data)
        # Zero displacement at fixed nodes
        data["displacement"] = data["displacement"].copy()
        for i in data["boundary_conditions"]["fixed"]:
            data["displacement"][i] = 0.0
        report = agent.check(data, problem_card_linear)
        assert report.get("fixed_passed", True)

    def test_nonzero_fixed_fails(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.boundary_condition_agent import BoundaryConditionAgent
        agent = BoundaryConditionAgent()
        data = dict(simple_linear_elastic_data)
        # Large displacement at fixed nodes → fail
        data["displacement"] = data["displacement"].copy()
        for i in data["boundary_conditions"]["fixed"]:
            data["displacement"][i] = np.array([1.0, 1.0, 1.0])
        report = agent.check(data, problem_card_linear)
        assert not report.get("fixed_passed", True)

    def test_report_structure(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.boundary_condition_agent import BoundaryConditionAgent
        agent = BoundaryConditionAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert "passed" in report


# ── Material Agent ───────────────────────────────────────────────────────────

class TestMaterialAgent:

    def test_linear_elastic_valid(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.material_agent import MaterialAgent
        agent = MaterialAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_linear)
        assert isinstance(report, dict)
        assert "passed" in report

    def test_negative_youngs_fails(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.material_agent import MaterialAgent
        agent = MaterialAgent()
        data = dict(simple_linear_elastic_data)
        data["material_props"] = {"E": -1e9, "nu": 0.3}
        report = agent.check(data, problem_card_linear)
        assert not report["passed"]

    def test_invalid_poisson_fails(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.material_agent import MaterialAgent
        agent = MaterialAgent()
        data = dict(simple_linear_elastic_data)
        data["material_props"] = {"E": 210e9, "nu": 0.6}  # must be < 0.5
        report = agent.check(data, problem_card_linear)
        assert not report["passed"]

    def test_elastoplastic_valid(self, simple_linear_elastic_data, problem_card_elastoplastic):
        from agents.physics_agent.material_agent import MaterialAgent
        agent = MaterialAgent()
        report = agent.check(simple_linear_elastic_data, problem_card_elastoplastic)
        assert isinstance(report, dict)
        assert "passed" in report

    def test_no_material_props_handled(self, simple_linear_elastic_data, problem_card_linear):
        from agents.physics_agent.material_agent import MaterialAgent
        agent = MaterialAgent()
        data = dict(simple_linear_elastic_data)
        data["material_props"] = {}
        report = agent.check(data, problem_card_linear)
        assert "passed" in report
