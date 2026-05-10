"""
RAGRetriever — central retrieval interface used by all agents.

Wraps EmbeddingService + RAGVectorStore and exposes 5 domain-specific
query methods. Gracefully returns empty lists when FAISS or
sentence-transformers are unavailable, so agents fall back to SQL.
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from agents.rag.embedding_service import EmbeddingService
from agents.rag.vector_store import RAGVectorStore
from agents.rag.document_builder import (
    build_run_doc, build_run_metadata,
    build_failure_doc, build_failure_metadata,
    build_model_perf_doc, build_model_perf_metadata,
    build_lambda_doc, build_lambda_metadata,
    build_dna_doc, build_dna_metadata,
)


class RAGRetriever:
    """
    Single shared instance, created once by MasterOrchestrator and
    injected into every agent + RunDatabase.
    """

    def __init__(self, config: dict):
        rag_cfg = config.get("rag", {})
        self.enabled      = rag_cfg.get("enabled", True)
        self.min_sim      = rag_cfg.get("min_similarity", 0.55)
        self.top_k        = rag_cfg.get("top_k_default", 5)
        index_dir         = rag_cfg.get("index_dir", "memory/rag_indices")

        self.embedder = EmbeddingService()
        self.store    = RAGVectorStore(index_dir)

        if self.enabled and self.store.available and self.embedder.available:
            logger.info(f"RAGRetriever ready — indices: {self.store.stats()}")
        else:
            logger.info("RAGRetriever: running in SQL-only mode (FAISS/transformers not installed)")

    @property
    def ready(self) -> bool:
        return self.enabled and self.store.available and self.embedder.available

    # ── Indexing (called by RunDatabase on every save) ─────────────

    def index_run(self, run_record):
        """Index a completed run into geometry_index and physics_model_index."""
        if not self.ready:
            return
        try:
            # geometry_index
            doc = build_run_doc(run_record)
            vec = self.embedder.embed(doc)
            if vec is not None:
                self.store.add("geometry_index", vec, build_run_metadata(run_record))

            # physics_model_index
            pm_doc = build_model_perf_doc(
                run_record.model_used, run_record.physics_type, run_record.mesh_type
            )
            pm_vec = self.embedder.embed(pm_doc)
            if pm_vec is not None:
                self.store.add(
                    "physics_model_index", pm_vec,
                    build_model_perf_metadata(
                        run_record.model_used, run_record.physics_type,
                        run_record.mesh_type, run_record.r2_score,
                        bool(run_record.success), run_record.data_size,
                    )
                )
        except Exception as e:
            logger.warning(f"RAG index_run failed: {e}")

    def index_failure(self, failure_record, r2_after: float = 0.0,
                       failed_checks: Optional[List[str]] = None,
                       lambda_json: Optional[dict] = None):
        """Index a failure into failure_index and optionally lambda_index."""
        if not self.ready:
            return
        try:
            doc = build_failure_doc(failure_record)
            vec = self.embedder.embed(doc)
            if vec is not None:
                self.store.add(
                    "failure_index", vec,
                    build_failure_metadata(failure_record, r2_after)
                )

            # Also index lambda state if fix improved R²
            if lambda_json and failed_checks and r2_after > failure_record.r2_at_failure:
                lam_doc = build_lambda_doc(
                    failure_record.physics_type, failed_checks, lambda_json
                )
                lam_vec = self.embedder.embed(lam_doc)
                if lam_vec is not None:
                    self.store.add(
                        "lambda_index", lam_vec,
                        build_lambda_metadata(
                            failure_record.physics_type, failed_checks,
                            lambda_json, r2_after, failure_record.run_id,
                        )
                    )
        except Exception as e:
            logger.warning(f"RAG index_failure failed: {e}")

    def index_custom_model(self, model_id: str, name: str, dna_dict: dict,
                            code: str, physics_type: str, r2: float,
                            failed_checks: Optional[List[str]] = None):
        """Index a custom model into custom_dna_index."""
        if not self.ready:
            return
        try:
            checks = failed_checks or []
            doc = build_dna_doc(dna_dict, physics_type, checks)
            vec = self.embedder.embed(doc)
            if vec is not None:
                self.store.add(
                    "custom_dna_index", vec,
                    build_dna_metadata(model_id, name, dna_dict, code,
                                       physics_type, r2, checks)
                )
        except Exception as e:
            logger.warning(f"RAG index_custom_model failed: {e}")

    # ── Retrieval ─────────────────────────────────────────────────

    def find_similar_problems(self, problem_card,
                               top_k: Optional[int] = None,
                               success_only: bool = False) -> List[dict]:
        """
        Top-k most similar past runs by problem description.
        Returns list of dicts with keys: similarity, run_id, model, r2, success, ...
        """
        if not self.ready:
            return []
        try:
            vec = self.embedder.embed_problem(problem_card)
            if vec is None:
                return []
            k = top_k or self.top_k
            filter_fn = (lambda m: m.get("success", False)) if success_only else None
            results = self.store.search(
                "geometry_index", vec, top_k=k,
                min_similarity=self.min_sim, filter_fn=filter_fn
            )
            logger.debug(f"RAG find_similar_problems: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"RAG find_similar_problems failed: {e}")
            return []

    def find_fixes_for_failure(self, model_name: str, failure_reason: str,
                                physics_type: str,
                                top_k: Optional[int] = None) -> List[dict]:
        """
        Top-k similar past failures and the fix that was applied.
        Returns list of dicts with keys: similarity, model_name, failure_reason,
        fix_tried, r2_before, r2_after.
        """
        if not self.ready:
            return []
        try:
            vec = self.embedder.embed_failure(
                model_name, failure_reason, "", physics_type
            )
            if vec is None:
                return []
            k = top_k or self.top_k
            results = self.store.search(
                "failure_index", vec, top_k=k, min_similarity=self.min_sim
            )
            logger.debug(f"RAG find_fixes_for_failure: {len(results)} results for "
                         f"{model_name}+{failure_reason[:40]}")
            return results
        except Exception as e:
            logger.warning(f"RAG find_fixes_for_failure failed: {e}")
            return []

    def find_model_history(self, model_name: str, problem_card,
                            top_k: Optional[int] = None) -> List[dict]:
        """
        Top-k performance records for this model on similar problems.
        Returns list of dicts with keys: similarity, model_name, r2, success, physics_type.
        """
        if not self.ready:
            return []
        try:
            physics_type = str(getattr(problem_card, "physics_type", "unknown"))
            mesh_type    = str(getattr(problem_card, "mesh_type", "unknown"))
            vec = self.embedder.embed_model_perf(model_name, physics_type, mesh_type)
            if vec is None:
                return []
            k = top_k or min(self.top_k * 2, 10)
            filter_fn = lambda m: m.get("model_name") == model_name
            results = self.store.search(
                "physics_model_index", vec, top_k=k,
                min_similarity=self.min_sim, filter_fn=filter_fn
            )
            logger.debug(f"RAG find_model_history: {len(results)} records for {model_name}")
            return results
        except Exception as e:
            logger.warning(f"RAG find_model_history failed: {e}")
            return []

    def find_lambda_history(self, physics_type: str,
                             failed_checks: List[str],
                             top_k: Optional[int] = None) -> List[dict]:
        """
        Top-k successful lambda configs for similar (physics, failure) combinations.
        Returns list of dicts with keys: similarity, lambda_json, r2, physics_type.
        """
        if not self.ready:
            return []
        try:
            dummy_lambdas: Dict[str, float] = {}
            vec = self.embedder.embed_lambda_state(
                physics_type, failed_checks, dummy_lambdas
            )
            if vec is None:
                return []
            k = top_k or 3
            results = self.store.search(
                "lambda_index", vec, top_k=k, min_similarity=self.min_sim
            )
            logger.debug(f"RAG find_lambda_history: {len(results)} results for "
                         f"{physics_type}+failed={failed_checks}")
            return results
        except Exception as e:
            logger.warning(f"RAG find_lambda_history failed: {e}")
            return []

    def find_similar_custom_dna(self, physics_type: str,
                                 failed_checks: List[str],
                                 top_k: Optional[int] = None,
                                 min_r2: float = 0.90) -> List[dict]:
        """
        Top-k successful custom model DNA patterns for similar problems.
        Returns list of dicts with keys: similarity, name, core_blocks, r2, code (snippet).
        """
        if not self.ready:
            return []
        try:
            vec = self.embedder.embed_dna({}, physics_type, failed_checks)
            if vec is None:
                return []
            k = top_k or 3
            filter_fn = lambda m: m.get("r2", 0.0) >= min_r2
            results = self.store.search(
                "custom_dna_index", vec, top_k=k,
                min_similarity=self.min_sim, filter_fn=filter_fn
            )
            logger.debug(f"RAG find_similar_custom_dna: {len(results)} results "
                         f"(physics={physics_type})")
            return results
        except Exception as e:
            logger.warning(f"RAG find_similar_custom_dna failed: {e}")
            return []

    def get_stats(self) -> dict:
        return {
            "ready":   self.ready,
            "indices": self.store.stats(),
        }
