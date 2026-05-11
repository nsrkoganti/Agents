"""
FAISS-backed vector store with 5 named indices and disk persistence.

Each index is a flat inner-product index (IndexFlatIP).
Because vectors are L2-normalised before insertion, inner product == cosine similarity.

Indices:
  geometry_index     — past runs keyed by problem description
  failure_index      — past failures keyed by failure signature
  physics_model_index— model performance keyed by (model, physics, mesh)
  lambda_index       — lambda tuning sessions keyed by problem + failed checks
  custom_dna_index   — custom model DNA keyed by physics failure signature
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from loguru import logger


INDICES = [
    "geometry_index",
    "failure_index",
    "physics_model_index",
    "lambda_index",
    "custom_dna_index",
]
EMBEDDING_DIM = 384


class RAGVectorStore:
    """
    Manages 5 named FAISS indices with metadata sidecar files.
    Thread-safety: not guaranteed — single-process use only.
    """

    def __init__(self, index_dir: str = "memory/rag_indices"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._indices: Dict[str, Any] = {}      # name -> faiss.Index
        self._metadata: Dict[str, List[dict]] = {}  # name -> list of dicts
        self._faiss_available = False
        self._load_all()

    def _load_all(self):
        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
        except ImportError:
            logger.warning(
                "faiss-cpu not installed. RAG vector search disabled. "
                "Install with: pip install faiss-cpu"
            )
            return

        for name in INDICES:
            faiss_path = self.index_dir / f"{name}.faiss"
            meta_path  = self.index_dir / f"{name}.meta.pkl"

            if faiss_path.exists() and meta_path.exists():
                try:
                    self._indices[name]  = faiss.read_index(str(faiss_path))
                    with open(meta_path, "rb") as f:
                        self._metadata[name] = pickle.load(f)
                    logger.debug(
                        f"RAG: loaded {name} ({self._indices[name].ntotal} vectors)"
                    )
                except Exception as e:
                    logger.warning(f"RAG: failed to load {name}: {e} — creating fresh")
                    self._create_fresh(name)
            else:
                self._create_fresh(name)

    def _create_fresh(self, name: str):
        if not self._faiss_available:
            return
        self._indices[name]  = self._faiss.IndexFlatIP(EMBEDDING_DIM)
        self._metadata[name] = []

    def _save(self, name: str):
        if not self._faiss_available or name not in self._indices:
            return
        try:
            self._faiss.write_index(
                self._indices[name],
                str(self.index_dir / f"{name}.faiss")
            )
            with open(self.index_dir / f"{name}.meta.pkl", "wb") as f:
                pickle.dump(self._metadata[name], f)
        except Exception as e:
            logger.warning(f"RAG: failed to persist {name}: {e}")

    # ── Public API ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._faiss_available

    def add(self, index_name: str, vector: np.ndarray, metadata: dict):
        """
        Add one vector with associated metadata dict.
        Vector must be L2-normalised float32 shape (384,).
        """
        if not self._faiss_available or index_name not in self._indices:
            return
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        self._indices[index_name].add(vec)
        self._metadata[index_name].append(metadata)
        self._save(index_name)

    def search(
        self,
        index_name: str,
        vector: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> List[dict]:
        """
        Cosine similarity search. Returns list of dicts:
          {"similarity": float, "metadata": dict, ...metadata fields flattened}
        Results with similarity < min_similarity are excluded.
        Optional filter_fn(metadata) -> bool applied after retrieval.
        """
        if not self._faiss_available or index_name not in self._indices:
            return []
        index = self._indices[index_name]
        if index.ntotal == 0:
            return []

        k = min(top_k * 3, index.ntotal)  # over-fetch to allow post-filtering
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        similarities, ids = index.search(vec, k)

        results = []
        for sim, idx in zip(similarities[0], ids[0]):
            if idx < 0 or float(sim) < min_similarity:
                continue
            meta = self._metadata[index_name][idx]
            if filter_fn and not filter_fn(meta):
                continue
            entry = {"similarity": float(sim), "metadata": meta}
            entry.update(meta)
            results.append(entry)
            if len(results) >= top_k:
                break

        return results

    def count(self, index_name: str) -> int:
        if index_name not in self._indices:
            return 0
        return self._indices[index_name].ntotal

    def stats(self) -> Dict[str, int]:
        return {name: self.count(name) for name in INDICES}
