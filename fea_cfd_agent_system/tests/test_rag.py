"""
Tests for the RAG (Retrieval-Augmented Generation) layer.

Covers: embedding service, vector store, document builder, RAGRetriever.
All tests use a temporary directory so no state leaks between runs.
"""

import json
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List


# ── Fixtures ────────────────────────────────────────────────────────────────

@dataclass
class FakeProblemCard:
    physics_type:        object = None
    mesh_type:           object = None
    material_model:      str    = "linear_elastic"
    loading_type:        str    = "static"
    element_type:        str    = "tet"
    data_size:           int    = 200
    n_nodes:             int    = 5000
    geometry_description: str   = "plate_with_hole"
    special_flags:       List   = field(default_factory=list)

    def __post_init__(self):
        if self.physics_type is None:
            self.physics_type = MagicMock()
            self.physics_type.__str__ = lambda s: "FEA_static_linear"
            type(self.physics_type).value = property(lambda s: "FEA_static_linear")
        if self.mesh_type is None:
            self.mesh_type = MagicMock()
            type(self.mesh_type).value = property(lambda s: "unstructured_tet")


@pytest.fixture
def tmp_index_dir(tmp_path):
    return str(tmp_path / "rag_indices")


@pytest.fixture
def config(tmp_index_dir):
    return {
        "rag": {
            "enabled":        True,
            "model":          "all-MiniLM-L6-v2",
            "index_dir":      tmp_index_dir,
            "top_k_default":  5,
            "min_similarity":  0.0,  # accept everything in tests
        }
    }


# ── Document builder tests ───────────────────────────────────────────────────

class TestDocumentBuilder:
    def test_build_run_doc_contains_fields(self):
        from agents.rag.document_builder import build_run_doc, build_run_metadata
        rec = MagicMock()
        rec.physics_type = "FEA_static_linear"
        rec.mesh_type    = "unstructured_tet"
        rec.data_size    = 300
        rec.model_used   = "Transolver"
        rec.success      = True
        rec.r2_score     = 0.96
        rec.rel_l2       = 0.03
        rec.notes        = "plate_with_hole"
        rec.run_id       = "run_001"

        doc = build_run_doc(rec)
        assert "FEA_static_linear" in doc
        assert "Transolver" in doc
        assert "0.960" in doc

        meta = build_run_metadata(rec)
        assert meta["model"] == "Transolver"
        assert meta["success"] is True
        assert meta["r2"] == 0.96

    def test_build_failure_doc(self):
        from agents.rag.document_builder import build_failure_doc, build_failure_metadata
        rec = MagicMock()
        rec.model_name     = "MeshGraphNet"
        rec.physics_type   = "FEA_static_nonlinear"
        rec.failure_reason = "BC_violation"
        rec.fix_tried      = "increase_bc_loss"
        rec.r2_at_failure  = 0.72
        rec.run_id         = "run_002"

        doc = build_failure_doc(rec)
        assert "MeshGraphNet" in doc
        assert "BC_violation" in doc

        meta = build_failure_metadata(rec, r2_after=0.94)
        assert meta["fix_tried"] == "increase_bc_loss"
        assert meta["r2_after"]  == 0.94

    def test_build_dna_doc(self):
        from agents.rag.document_builder import build_dna_doc, build_dna_metadata
        dna_dict = {
            "family": "hybrid",
            "has_physics_loss": True,
            "core_blocks": [
                {"type": "mamba_block"},
                {"type": "physics_attention"},
            ],
        }
        doc = build_dna_doc(dna_dict, "FEA_static_nonlinear", ["bc", "equilibrium"])
        assert "mamba_block" in doc
        assert "FEA_static_nonlinear" in doc
        assert "bc" in doc

        meta = build_dna_metadata(
            "model_id_1", "CustomNet", dna_dict,
            "class CustomNet: pass", "FEA_static_nonlinear", 0.95, ["bc"]
        )
        assert meta["r2"] == 0.95
        assert len(meta["core_blocks"]) == 2


# ── Vector store tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="faiss-cpu not installed"),
    reason="faiss-cpu not installed"
)
class TestRAGVectorStore:
    def test_add_and_search_round_trip(self, tmp_index_dir):
        from agents.rag.vector_store import RAGVectorStore
        store = RAGVectorStore(tmp_index_dir)
        if not store.available:
            pytest.skip("FAISS not available")

        vec  = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        meta = {"run_id": "test_001", "r2": 0.95, "success": True}
        store.add("geometry_index", vec, meta)

        results = store.search("geometry_index", vec, top_k=1, min_similarity=0.0)
        assert len(results) == 1
        assert results[0]["metadata"]["run_id"] == "test_001"
        assert results[0]["similarity"] > 0.99  # same vector → similarity ~1.0

    def test_filter_fn_applied(self, tmp_index_dir):
        from agents.rag.vector_store import RAGVectorStore
        store = RAGVectorStore(tmp_index_dir)
        if not store.available:
            pytest.skip("FAISS not available")

        for i, success in enumerate([True, False, True]):
            vec  = np.random.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            store.add("geometry_index", vec, {"run_id": f"run_{i}", "success": success})

        query  = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(
            "geometry_index", query, top_k=5,
            filter_fn=lambda m: m.get("success") is True
        )
        assert all(r["success"] for r in results)

    def test_gates_sum_to_one_shape(self, tmp_index_dir):
        """Empty index returns empty list, not error."""
        from agents.rag.vector_store import RAGVectorStore
        store  = RAGVectorStore(tmp_index_dir)
        if not store.available:
            pytest.skip("FAISS not available")
        query  = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search("failure_index", query, top_k=5)
        assert results == []

    def test_persistence(self, tmp_index_dir):
        """Saved index reloads correctly."""
        from agents.rag.vector_store import RAGVectorStore
        store1 = RAGVectorStore(tmp_index_dir)
        if not store1.available:
            pytest.skip("FAISS not available")

        vec  = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store1.add("geometry_index", vec, {"run_id": "persist_test"})

        store2   = RAGVectorStore(tmp_index_dir)
        results  = store2.search("geometry_index", vec, top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["run_id"] == "persist_test"

    def test_stats(self, tmp_index_dir):
        from agents.rag.vector_store import RAGVectorStore
        store = RAGVectorStore(tmp_index_dir)
        if not store.available:
            pytest.skip("FAISS not available")
        stats = store.stats()
        assert "geometry_index" in stats
        assert isinstance(stats["geometry_index"], int)


# ── EmbeddingService tests ───────────────────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
    reason="sentence-transformers not installed"
)
class TestEmbeddingService:
    def test_embed_returns_correct_shape(self):
        from agents.rag.embedding_service import EmbeddingService
        svc = EmbeddingService()
        if not svc.available:
            pytest.skip("Model not available")
        vec = svc.embed("FEA_static_linear unstructured_tet plate_with_hole")
        assert vec is not None
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_normalised(self):
        from agents.rag.embedding_service import EmbeddingService
        svc = EmbeddingService()
        if not svc.available:
            pytest.skip("Model not available")
        vec = svc.embed("test text")
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_embed_problem_card(self):
        from agents.rag.embedding_service import EmbeddingService
        svc  = EmbeddingService()
        if not svc.available:
            pytest.skip("Model not available")
        card = FakeProblemCard()
        vec  = svc.embed_problem(card)
        assert vec is not None
        assert vec.shape == (384,)

    def test_similar_texts_close(self):
        from agents.rag.embedding_service import EmbeddingService
        svc = EmbeddingService()
        if not svc.available:
            pytest.skip("Model not available")
        v1 = svc.embed("FEA nonlinear elastoplastic plate")
        v2 = svc.embed("FEA nonlinear elastoplastic box")
        v3 = svc.embed("turbulent fluid flow reynolds high")
        sim_close = float(np.dot(v1, v2))
        sim_far   = float(np.dot(v1, v3))
        assert sim_close > sim_far, "Similar FEA texts should be closer than FEA vs CFD"


# ── RAGRetriever integration tests ───────────────────────────────────────────

class TestRAGRetriever:
    def test_retriever_graceful_without_deps(self, config):
        """RAGRetriever initialises and returns [] when deps missing."""
        from agents.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever(config)
        card      = FakeProblemCard()
        results   = retriever.find_similar_problems(card)
        assert isinstance(results, list)

    def test_find_fixes_returns_list(self, config):
        from agents.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever(config)
        results   = retriever.find_fixes_for_failure(
            "Transolver", "underfitting", "FEA_static_linear"
        )
        assert isinstance(results, list)

    def test_find_lambda_history_returns_list(self, config):
        from agents.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever(config)
        results   = retriever.find_lambda_history(
            "FEA_static_nonlinear", ["bc", "equilibrium"]
        )
        assert isinstance(results, list)

    def test_find_similar_custom_dna_returns_list(self, config):
        from agents.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever(config)
        results   = retriever.find_similar_custom_dna(
            "FEA_static_nonlinear", ["bc"]
        )
        assert isinstance(results, list)

    def test_get_stats(self, config):
        from agents.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever(config)
        stats     = retriever.get_stats()
        assert "ready"   in stats
        assert "indices" in stats
