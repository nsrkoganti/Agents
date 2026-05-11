"""
Embedding service — wraps sentence-transformers to produce 384-dim vectors
for all text documents in the RAG layer.

Uses all-MiniLM-L6-v2 (local, ~90 MB, no API key required).
Vectors are L2-normalised before storage so inner product == cosine similarity.
"""

import numpy as np
from loguru import logger
from typing import Optional, List


class EmbeddingService:
    """
    Singleton-friendly embedding service.
    Call embed() or the convenience methods to get float32 numpy vectors.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(f"EmbeddingService: loaded {self.MODEL_NAME}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "RAG will fall back to SQL-only retrieval. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Embed a single string.
        Returns L2-normalised float32 vector of shape (384,), or None if unavailable.
        """
        self._load()
        if self._model is None:
            return None
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed multiple strings at once. Returns (N, 384) or None."""
        self._load()
        if self._model is None:
            return None
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    # ── Convenience builders ──────────────────────────────────────

    def embed_problem(self, problem_card) -> Optional[np.ndarray]:
        """Canonical embedding for a ProblemCard."""
        text = (
            f"physics={getattr(problem_card, 'physics_type', 'unknown')} "
            f"mesh={getattr(problem_card, 'mesh_type', 'unknown')} "
            f"material={getattr(problem_card, 'material_model', 'unknown')} "
            f"loading={getattr(problem_card, 'loading_type', 'unknown')} "
            f"element={getattr(problem_card, 'element_type', 'unknown')} "
            f"data_size={getattr(problem_card, 'data_size', 0)} "
            f"n_nodes={getattr(problem_card, 'n_nodes', 0)} "
            f"geometry={getattr(problem_card, 'geometry_description', '')} "
            f"flags={' '.join(getattr(problem_card, 'special_flags', []))}"
        )
        return self.embed(text)

    def embed_failure(self, model_name: str, failure_reason: str,
                       fix_tried: str, physics_type: str) -> Optional[np.ndarray]:
        """Canonical embedding for a failure record."""
        text = (
            f"model={model_name} "
            f"physics={physics_type} "
            f"failure={failure_reason} "
            f"fix={fix_tried}"
        )
        return self.embed(text)

    def embed_lambda_state(self, physics_type: str,
                            failed_checks: List[str],
                            lambda_json: dict) -> Optional[np.ndarray]:
        """Canonical embedding for a lambda tuning record."""
        checks = " ".join(failed_checks) if failed_checks else "none"
        lambdas = " ".join(f"{k}={v:.2f}" for k, v in sorted(lambda_json.items()))
        text = f"physics={physics_type} failed_checks={checks} lambdas={lambdas}"
        return self.embed(text)

    def embed_dna(self, dna_dict: dict, physics_type: str,
                   failed_checks: List[str]) -> Optional[np.ndarray]:
        """Canonical embedding for a custom model DNA."""
        blocks = " ".join(
            b.get("type", "") for b in dna_dict.get("core_blocks", [])
        )
        checks = " ".join(failed_checks) if failed_checks else "none"
        text = (
            f"physics={physics_type} "
            f"failed={checks} "
            f"blocks={blocks} "
            f"family={dna_dict.get('family', '')} "
            f"has_physics_loss={dna_dict.get('has_physics_loss', False)}"
        )
        return self.embed(text)

    def embed_model_perf(self, model_name: str, physics_type: str,
                          mesh_type: str) -> Optional[np.ndarray]:
        """Canonical embedding for a model performance record."""
        text = (
            f"model={model_name} "
            f"physics={physics_type} "
            f"mesh={mesh_type}"
        )
        return self.embed(text)
