"""Tests for the self-learning system."""

import pytest
import tempfile
import os
import datetime
from memory.run_database import RunDatabase, RunRecord, FailureRecord


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db = RunDatabase(db_path=path)
    yield db
    os.unlink(path)


def _make_run(run_id, model, success, physics_type, mesh_type, r2=0.9):
    return RunRecord(
        run_id=run_id,
        physics_type=physics_type,
        mesh_type=mesh_type,
        data_size=5000,
        model_used=model,
        r2_score=r2,
        rel_l2=0.05,
        success=success,
        n_iterations=3,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )


def test_knowledge_base_recommendation(tmp_db):
    from agents.self_learning.knowledge_base import KnowledgeBase
    from agents.orchestrator.agent_state import (
        AgentSystemState, PhysicsType, MeshType, ProblemCard
    )

    phys_val = PhysicsType.CFD_INCOMPRESSIBLE.value
    mesh_val = MeshType.UNSTRUCTURED_POLY.value

    tmp_db.save_model_performance(
        "Transolver", phys_val, mesh_val,
        r2=0.95, rel_l2=0.03, success=True, data_size=5000
    )
    tmp_db.save_model_performance(
        "Transolver", phys_val, mesh_val,
        r2=0.93, rel_l2=0.04, success=True, data_size=5000
    )

    config = {"db_path": str(tmp_db.db_path)}
    kb     = KnowledgeBase(config, tmp_db)

    state = AgentSystemState(data_path="dummy.vtk")
    state.problem_card = ProblemCard(
        physics_type=PhysicsType.CFD_INCOMPRESSIBLE,
        mesh_type=MeshType.UNSTRUCTURED_POLY,
    )

    rec = kb.get_model_recommendation(state)
    assert rec is not None
    assert rec.get("model_name") == "Transolver"


def test_success_rate_tracking(tmp_db):
    from agents.orchestrator.agent_state import PhysicsType, MeshType

    phys = PhysicsType.CFD_INCOMPRESSIBLE.value
    mesh = MeshType.UNSTRUCTURED_POLY.value

    for i in range(5):
        tmp_db.save_run(_make_run(f"s{i}", "Transolver", True,  phys, mesh, 0.94))
    for i in range(5):
        tmp_db.save_run(_make_run(f"f{i}", "Transolver", False, phys, mesh, 0.40))

    rate = tmp_db.get_success_rate(phys)
    assert 0.4 < rate < 0.6


def test_self_learning_updater(tmp_db):
    from agents.self_learning.self_learning_updater import SelfLearningUpdater
    from agents.orchestrator.agent_state import (
        AgentSystemState, PhysicsType, MeshType, ProblemCard, EvaluationResult
    )

    updater = SelfLearningUpdater({"db_path": str(tmp_db.db_path)}, tmp_db)

    state = AgentSystemState(data_path="dummy.vtk")
    state.problem_card = ProblemCard(
        physics_type=PhysicsType.CFD_INCOMPRESSIBLE,
        mesh_type=MeshType.UNSTRUCTURED_POLY,
    )
    state.pipeline_success  = True
    state.current_attempt   = 3
    state.evaluation_result = EvaluationResult(
        r2_score=0.94, rel_l2_error=0.03,
        max_point_error=0.08, passed=True,
    )

    updater.update(state)

    runs = tmp_db.get_recent_runs(5)
    assert len(runs) >= 1
    assert runs[0]["r2_score"] == pytest.approx(0.94)
