"""
Tests for the Dataset Agent subsystem:
- DatasetSearchAgent (curated filter + LLM ranking stub)
- DatasetDownloadAgent (cache hit / miss logic)
- DatasetValidatorAgent (NPZ, CSV, format detection, license check)
- DatasetOrchestrator (full pipeline with mocked sub-agents)
- Inter-agent message bus (REQUEST_MORE_DATA, INSUFFICIENT_BC_DATA)
"""

import os
import json
import datetime
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from agents.dataset_agent.dataset_searcher import DatasetSearchAgent, CURATED_DATASETS
from agents.dataset_agent.dataset_downloader import DatasetDownloadAgent
from agents.dataset_agent.dataset_validator import DatasetValidatorAgent
from agents.dataset_agent.dataset_orchestrator import DatasetOrchestrator
from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, ProblemCard,
    PhysicsType, MeshType, ProblemType,
)


# ── Helpers ────────────────────────────────────────────────────────────────

CONFIG = {
    "dataset_cache_dir": "/tmp/test_dataset_cache",
    "llm": {"dev_provider": "anthropic"},
}


def _make_state(search_datasets: bool = False, data_path: str = "") -> AgentSystemState:
    state = AgentSystemState(data_path=data_path, search_datasets=search_datasets)
    state.problem_card = ProblemCard(
        problem_type=ProblemType.FIELD_REGRESSION,
        physics_type=PhysicsType.CFD_INCOMPRESSIBLE,
        mesh_type=MeshType.UNSTRUCTURED_POLY,
        data_size=500,
    )
    return state


# ── DatasetSearchAgent ─────────────────────────────────────────────────────

class TestDatasetSearchAgent:

    def setup_method(self):
        with patch("agents.dataset_agent.dataset_searcher.get_dev_llm"):
            self.agent = DatasetSearchAgent(CONFIG)

    def test_filter_curated_cfd_incompressible(self):
        results = self.agent._filter_curated("CFD_incompressible", "unstructured_polyhedral")
        names = [d["name"] for d in results]
        assert len(results) >= 1
        assert any("AirfRANS" in n or "CFDBench" in n or "PhysicsNeMo" in n or "Transport" in n
                   for n in names)

    def test_filter_curated_fallback(self):
        """Unknown physics type falls back to top-3 curated."""
        results = self.agent._filter_curated("unknown_physics", "structured")
        assert len(results) == 3  # fallback always returns top 3

    def test_physics_to_query_mapping(self):
        # "cfd_incompressible" hits the "cfd" key first → returns generic CFD query
        q_cfd = self.agent._physics_to_query("cfd_incompressible").lower()
        assert "cfd" in q_cfd or "fluid" in q_cfd or "incompressible" in q_cfd
        # turbulent type
        q_turb = self.agent._physics_to_query("CFD_incompressible_turbulent").lower()
        assert "turbulent" in q_turb or "rans" in q_turb or "cfd" in q_turb
        # FEA type
        q = self.agent._physics_to_query("FEA_static_linear").lower()
        assert "fea" in q or "finite element" in q or "structural" in q

    def test_rank_with_llm_single_dataset(self):
        datasets = [CURATED_DATASETS[0].copy()]
        result = self.agent._rank_with_llm(datasets, "cfd", "unstructured", 100, "")
        assert result == datasets

    def test_rank_with_llm_fallback_on_error(self):
        """LLM failure returns original order."""
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.side_effect = RuntimeError("LLM down")
        datasets = list(CURATED_DATASETS[:3])
        result = self.agent._rank_with_llm(datasets, "cfd", "unstructured", 100, "")
        assert result == datasets

    def test_search_huggingface_handles_network_error(self):
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            result = self.agent._search_huggingface("CFD_incompressible")
        assert result == []

    def test_search_zenodo_handles_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("requests.get", return_value=mock_resp):
            result = self.agent._search_zenodo("CFD_incompressible")
        assert result == []

    def test_search_deduplicates(self):
        """Duplicate repo_ids should be removed."""
        with patch("agents.dataset_agent.dataset_searcher.get_dev_llm"):
            agent = DatasetSearchAgent(CONFIG)

        duplicate = dict(CURATED_DATASETS[0])

        with patch.object(agent, "_search_huggingface", return_value=[duplicate]), \
             patch.object(agent, "_search_zenodo", return_value=[]), \
             patch.object(agent, "_rank_with_llm", side_effect=lambda d, *a, **k: d):
            results = agent.search("CFD_incompressible", "unstructured", 100)

        names = [d["name"] for d in results]
        assert len(names) == len(set(names))


# ── DatasetDownloadAgent ───────────────────────────────────────────────────

class TestDatasetDownloadAgent:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        cfg = {**CONFIG, "dataset_cache_dir": self.tmp}
        self.agent = DatasetDownloadAgent(cfg)

    def test_cache_hit_skips_download(self):
        name = "MyDataset"
        local = Path(self.tmp) / name
        local.mkdir()
        (local / "data.csv").write_text("a,b\n1,2\n")

        result = self.agent.download({"name": name, "source": "huggingface", "repo_id": "x/y"})
        assert result == str(local)

    def test_get_cache_path_returns_none_when_empty(self):
        result = self.agent.get_cache_path({"name": "NonExistent"})
        assert result is None

    def test_get_cache_path_returns_path_when_exists(self):
        name = "CachedDataset"
        local = Path(self.tmp) / name
        local.mkdir()
        (local / "x.parquet").write_text("dummy")
        result = self.agent.get_cache_path({"name": name})
        assert result == str(local)

    def test_download_unknown_source_attempts_url(self):
        info = {"name": "TestDS", "source": "unknown", "download_url": None, "url": None}
        result = self.agent.download(info)
        assert result is None  # no URL → None

    def test_download_github_failure_returns_none(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="auth error")
            result = self.agent._download_github(
                {"repo_url": "https://github.com/fake/repo"},
                Path(self.tmp) / "repo"
            )
        assert result is None

    def test_download_url_streams_file(self):
        local = Path(self.tmp) / "StreamedDS"
        fake_content = b"col1,col2\n1.0,2.0\n3.0,4.0\n"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-length": str(len(fake_content))}
        mock_resp.iter_content = MagicMock(return_value=[fake_content])

        with patch("requests.get", return_value=mock_resp):
            result = self.agent._download_url(
                {"download_url": "http://example.com/data.csv", "url": None},
                local
            )
        assert result == str(local)
        assert (local / "data.csv").exists()


# ── DatasetValidatorAgent ──────────────────────────────────────────────────

class TestDatasetValidatorAgent:

    def setup_method(self):
        self.agent = DatasetValidatorAgent()
        self.tmp = tempfile.mkdtemp()

    def _write_npz(self, name="data.npz", **arrays):
        path = Path(self.tmp) / name
        np.savez(str(path), **arrays)
        return path

    def _write_csv(self, name="data.csv", n=20):
        import pandas as pd
        path = Path(self.tmp) / name
        df = pd.DataFrame({
            "velocity": np.random.rand(n),
            "pressure": np.random.rand(n),
            "x": np.random.rand(n),
        })
        df.to_csv(str(path), index=False)
        return path

    def test_validate_missing_path(self):
        report = self.agent.validate("/nonexistent/path", {"license": "mit"})
        assert report["valid"] is False
        assert any("does not exist" in i for i in report["issues"])

    def test_validate_blocked_license(self):
        report = self.agent.validate(self.tmp, {"license": "CC-BY-NC-4.0"})
        assert report["valid"] is False
        assert report["license_ok"] is False

    def test_validate_no_data_files(self):
        empty_dir = tempfile.mkdtemp()
        report = self.agent.validate(empty_dir, {"license": "mit"})
        assert report["valid"] is False
        assert any("No recognizable" in i for i in report["issues"])

    def test_validate_npz_with_physics_fields(self):
        n = 200
        self._write_npz(
            velocity=np.random.rand(n, 3),
            pressure=np.random.rand(n),
        )
        # Duplicate files to pass MIN_SAMPLES=5
        for i in range(6):
            self._write_npz(f"data_{i}.npz",
                            velocity=np.random.rand(n, 3),
                            pressure=np.random.rand(n))

        report = self.agent.validate(self.tmp, {"license": "apache-2.0"})
        assert report["valid"] is True
        assert "velocity" in report["fields_found"] or "pressure" in report["fields_found"]

    def test_validate_csv_with_physics_fields(self):
        for i in range(6):
            self._write_csv(f"data_{i}.csv", n=50)
        report = self.agent.validate(self.tmp, {"license": "mit"})
        assert report["valid"] is True
        assert "velocity" in report["fields_found"]

    def test_validate_too_few_samples(self):
        self._write_npz(velocity=np.random.rand(10, 3))
        report = self.agent.validate(self.tmp, {"license": "mit"})
        assert report["valid"] is False
        assert any("Too few" in i for i in report["issues"])

    def test_check_array_quality_flags_nan(self):
        result = {"ok": True, "issues": []}
        arr = np.full(100, np.nan)
        self.agent._check_array_quality(result, arr)
        assert result["ok"] is False
        assert any("NaN" in i for i in result["issues"])

    def test_check_array_quality_flags_inf(self):
        result = {"ok": True, "issues": []}
        arr = np.array([1.0, np.inf, 2.0])
        self.agent._check_array_quality(result, arr)
        assert any("Inf" in i for i in result["issues"])

    def test_detect_format(self):
        assert self.agent._detect_format(Path("x.vtu"))     == "vtu"
        assert self.agent._detect_format(Path("x.hdf5"))    == "hdf5"
        assert self.agent._detect_format(Path("x.parquet")) == "parquet"
        assert self.agent._detect_format(Path("x.npz"))     == "npz"
        assert self.agent._detect_format(Path("x.csv"))     == "csv"
        assert self.agent._detect_format(Path("x.nc"))      == "netcdf"

    def test_check_physics_fields_no_match(self):
        result = {"ok": True, "issues": []}
        self.agent._check_physics_fields(result, ["col_a", "col_b"])
        assert any("No recognizable physics" in i for i in result["issues"])

    def test_check_physics_fields_match(self):
        result = {"ok": True, "issues": []}
        self.agent._check_physics_fields(result, ["velocity", "pressure"])
        assert result["issues"] == []


# ── DatasetOrchestrator ────────────────────────────────────────────────────

class TestDatasetOrchestrator:

    def setup_method(self):
        with patch("agents.dataset_agent.dataset_searcher.get_dev_llm"):
            self.orchestrator = DatasetOrchestrator(CONFIG)

    def _make_valid_report(self):
        return {
            "valid": True, "format": "npz", "n_files": 10,
            "n_samples_estimated": 100, "fields_found": ["velocity", "pressure"],
            "license_ok": True, "issues": [], "local_path": "/tmp/fake",
        }

    def test_run_success(self):
        state = _make_state()
        fake_datasets = [
            {"name": "TestCFD", "source": "huggingface", "repo_id": "test/cfd",
             "license": "mit", "n_samples": 1000},
        ]
        with patch.object(self.orchestrator.searcher, "search", return_value=fake_datasets), \
             patch.object(self.orchestrator.downloader, "get_cache_path", return_value=None), \
             patch.object(self.orchestrator.downloader, "download", return_value="/tmp/fake"), \
             patch.object(self.orchestrator.validator, "validate", return_value=self._make_valid_report()):
            result = self.orchestrator.run(state)

        assert result.dataset_agent_status == AgentStatus.PASSED
        assert result.data_path == "/tmp/fake"
        assert result.selected_dataset["name"] == "TestCFD"
        assert any("DATASET_READY" in m["type"] for m in result.agent_messages)

    def test_run_no_datasets_found(self):
        state = _make_state()
        with patch.object(self.orchestrator.searcher, "search", return_value=[]):
            result = self.orchestrator.run(state)
        assert result.dataset_agent_status == AgentStatus.FAILED
        assert any("NO_DATASETS_FOUND" in m["type"] for m in result.agent_messages)

    def test_run_all_fail_validation(self):
        state = _make_state()
        bad_report = {"valid": False, "issues": ["bad data"], "format": "unknown", "n_files": 0}
        fake_datasets = [
            {"name": f"DS{i}", "source": "zenodo", "license": "mit"}
            for i in range(5)
        ]
        with patch.object(self.orchestrator.searcher, "search", return_value=fake_datasets), \
             patch.object(self.orchestrator.downloader, "get_cache_path", return_value=None), \
             patch.object(self.orchestrator.downloader, "download", return_value="/tmp/fake"), \
             patch.object(self.orchestrator.validator, "validate", return_value=bad_report):
            result = self.orchestrator.run(state)
        assert result.dataset_agent_status == AgentStatus.FAILED
        assert any("ALL_DATASETS_INVALID" in m["type"] for m in result.agent_messages)

    def test_run_uses_cache_hit(self):
        state = _make_state()
        fake_datasets = [
            {"name": "CachedDS", "source": "huggingface", "repo_id": "test/cached", "license": "mit"},
        ]
        with patch.object(self.orchestrator.searcher, "search", return_value=fake_datasets), \
             patch.object(self.orchestrator.downloader, "get_cache_path", return_value="/tmp/cached"), \
             patch.object(self.orchestrator.downloader, "download") as mock_dl, \
             patch.object(self.orchestrator.validator, "validate", return_value=self._make_valid_report()):
            result = self.orchestrator.run(state)

        mock_dl.assert_not_called()
        assert result.dataset_agent_status == AgentStatus.PASSED

    def test_run_skips_failed_download(self):
        state = _make_state()
        fake_datasets = [
            {"name": "BadDownload", "source": "url", "download_url": None, "license": "mit"},
            {"name": "GoodDS", "source": "huggingface", "repo_id": "test/good", "license": "mit"},
        ]
        call_count = [0]

        def fake_validate(path, info):
            call_count[0] += 1
            return self._make_valid_report()

        def fake_download(info):
            return None if info["name"] == "BadDownload" else "/tmp/good"

        with patch.object(self.orchestrator.searcher, "search", return_value=fake_datasets), \
             patch.object(self.orchestrator.downloader, "get_cache_path", return_value=None), \
             patch.object(self.orchestrator.downloader, "download", side_effect=fake_download), \
             patch.object(self.orchestrator.validator, "validate", side_effect=fake_validate):
            result = self.orchestrator.run(state)

        assert result.dataset_agent_status == AgentStatus.PASSED
        assert call_count[0] == 1  # only GoodDS validated

    def test_publish_message_format(self):
        state = _make_state()
        self.orchestrator._publish_message(state, "a", "b", "TEST_TYPE", "test reason")
        msg = state.agent_messages[-1]
        assert msg["from"] == "a"
        assert msg["to"] == "b"
        assert msg["type"] == "TEST_TYPE"
        assert msg["handled"] is False
        assert "timestamp" in msg


# ── Inter-agent message bus ────────────────────────────────────────────────

class TestInterAgentMessageBus:

    def test_iteration_agent_publishes_request_more_data(self):
        from agents.iteration_agent.iteration_agent import IterationAgent

        cfg = {"iteration": {"total_max_attempts": 24, "max_attempts_per_model": 3}}
        with patch("agents.iteration_agent.iteration_agent.get_dev_llm"):
            agent = IterationAgent(cfg)
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(
            content=json.dumps({"fix_type": "tune_hyperparameters", "fix_description": "lr"})
        )

        from agents.orchestrator.agent_state import (
            EvaluationResult, ModelCandidate, IterationRecord
        )
        state = _make_state()
        state.current_attempt = 5
        state.evaluation_result = EvaluationResult(r2_score=0.3, passed=False)
        state.selected_model = ModelCandidate(name="MLP")
        state.ranked_shortlist = [ModelCandidate(name="MLP")]

        result = agent.run(state)

        assert result.current_attempt == 6
        data_requests = [m for m in result.agent_messages if m["type"] == "REQUEST_MORE_DATA"]
        assert len(data_requests) == 1
        assert "R2=0.30" in data_requests[0]["reason"]

    def test_iteration_agent_no_duplicate_requests(self):
        from agents.iteration_agent.iteration_agent import IterationAgent
        from agents.orchestrator.agent_state import EvaluationResult, ModelCandidate

        cfg = {"iteration": {"total_max_attempts": 24, "max_attempts_per_model": 3}}
        with patch("agents.iteration_agent.iteration_agent.get_dev_llm"):
            agent = IterationAgent(cfg)
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = MagicMock(
            content=json.dumps({"fix_type": "tune_hyperparameters", "fix_description": "lr"})
        )

        state = _make_state()
        state.current_attempt = 5
        state.evaluation_result = EvaluationResult(r2_score=0.2, passed=False)
        state.selected_model = ModelCandidate(name="MLP")
        state.ranked_shortlist = [ModelCandidate(name="MLP")]
        state.agent_messages = [{
            "type": "REQUEST_MORE_DATA", "handled": False,
            "from": "iteration_agent", "to": "dataset_agent",
        }]

        result = agent.run(state)
        data_reqs = [m for m in result.agent_messages if m["type"] == "REQUEST_MORE_DATA"]
        assert len(data_reqs) == 1  # no duplicate

    def test_physics_agent_publishes_insufficient_bc(self):
        from agents.physics_agent.physics_master import PhysicsMasterAgent
        from agents.orchestrator.agent_state import PhysicsReport, TrainingResult

        cfg = {}
        agent = PhysicsMasterAgent(cfg)

        bc_result = {"passed": False, "failure_reason": "wall slip > threshold",
                     "missing_fields": ["wall_nodes"]}

        state = _make_state()
        state.agent_messages = []
        agent._publish_insufficient_bc(state, bc_result)

        msgs = [m for m in state.agent_messages if m["type"] == "INSUFFICIENT_BC_DATA"]
        assert len(msgs) == 1
        assert msgs[0]["from"] == "physics_agent"
        assert msgs[0]["to"] == "dataset_agent"
        assert "wall_nodes" in msgs[0]["missing"]

    def test_physics_agent_no_duplicate_bc_messages(self):
        from agents.physics_agent.physics_master import PhysicsMasterAgent

        agent = PhysicsMasterAgent({})
        state = _make_state()
        state.agent_messages = [{
            "type": "INSUFFICIENT_BC_DATA", "handled": False,
            "from": "physics_agent", "to": "dataset_agent",
        }]
        agent._publish_insufficient_bc(state, {"failure_reason": "x", "missing_fields": []})
        bc_msgs = [m for m in state.agent_messages if m["type"] == "INSUFFICIENT_BC_DATA"]
        assert len(bc_msgs) == 1  # no duplicate

    def test_agent_state_has_message_bus_fields(self):
        state = AgentSystemState()
        assert hasattr(state, "agent_messages")
        assert isinstance(state.agent_messages, list)
        assert hasattr(state, "search_datasets")
        assert hasattr(state, "discovered_datasets")
        assert hasattr(state, "selected_dataset")
        assert hasattr(state, "dataset_agent_status")
        assert state.dataset_agent_status == AgentStatus.PENDING
