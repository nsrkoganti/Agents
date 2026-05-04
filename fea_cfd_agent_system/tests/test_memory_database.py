"""Tests for the persistent memory database."""

import pytest
import tempfile
import os
from memory.run_database import RunDatabase, RunRecord, FailureRecord
import datetime


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db = RunDatabase(db_path=path)
    yield db
    os.unlink(path)


def test_save_and_retrieve_run(db):
    record = RunRecord(
        run_id="test_001",
        physics_type="cfd_incompressible",
        mesh_type="unstructured",
        data_size=10000,
        model_used="Transolver",
        r2_score=0.95,
        rel_l2=0.03,
        success=True,
        n_iterations=4,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    db.save_run(record)
    runs = db.get_recent_runs(n=5)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "test_001"
    assert runs[0]["r2_score"] == pytest.approx(0.95)


def test_failure_record(db):
    record = FailureRecord(
        run_id="test_002",
        model_name="FNO",
        failure_reason="mesh_incompatibility",
        fix_tried="switch_model",
        r2_at_failure=0.3,
        iteration=2,
        physics_type="cfd_incompressible",
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    db.save_failure(record)
    patterns = db.get_failure_patterns("cfd_incompressible")
    assert len(patterns) >= 1
    assert patterns[0]["model_name"] == "FNO"


def test_model_performance(db):
    db.save_model_performance(
        "Transolver", "cfd_incompressible", "unstructured",
        r2=0.94, rel_l2=0.04, success=True, data_size=5000
    )
    db.save_model_performance(
        "Transolver", "cfd_incompressible", "unstructured",
        r2=0.96, rel_l2=0.03, success=True, data_size=5000
    )
    best = db.get_best_model_for("cfd_incompressible", "unstructured")
    assert best is not None
    assert best["model_name"] == "Transolver"
    assert best["avg_r2"] > 0.9


def test_custom_model_save(db):
    dna = {"name": "TestModel", "family": "hybrid", "generation": 1}
    db.save_custom_model(
        model_id="custom_001",
        name="TestModel",
        dna=dna,
        code="import torch\n",
        problem="cfd_incompressible",
        r2=0.85,
        generation=1,
    )
    best = db.get_best_custom_model("cfd_incompressible")
    assert best is not None
    assert best["model_id"] == "custom_001"


def test_stats(db):
    stats = db.get_stats()
    assert "total_runs" in stats
    assert "success_rate" in stats
    assert 0.0 <= stats["success_rate"] <= 1.0
