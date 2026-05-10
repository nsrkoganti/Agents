"""Pattern recognizer — finds recurring success/failure patterns in run history.

Uses RAG vector search when available (semantic similarity).
Falls back to SQL LIKE queries when FAISS is not installed.
"""

import sqlite3
import json
from typing import List, Dict, Optional
from loguru import logger
from memory.run_database import RunDatabase


class PatternRecognizer:
    """Identifies patterns in failures and fixes that led to success."""

    def __init__(self, db: RunDatabase, retriever=None):
        self.db        = db
        self.retriever = retriever  # Optional[RAGRetriever]

    def find_lambda_patterns(self, physics_type: str) -> List[Dict]:
        """Find which lambda configurations succeeded for a physics type."""
        patterns = []
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute("""
                    SELECT f.lambda_used, r.r2_score, f.failure_type
                    FROM failures f
                    JOIN runs r ON f.run_id = r.run_id
                    WHERE r.physics_type = ? AND r.success = 1 AND f.lambda_used IS NOT NULL
                    ORDER BY r.r2_score DESC LIMIT 20
                """, (physics_type,)).fetchall()
                for row in rows:
                    try:
                        lambdas = json.loads(row[0])
                        patterns.append({
                            "lambda_weights": lambdas,
                            "r2_achieved":    row[1],
                            "failure_type":   row[2],
                        })
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Pattern recognition failed: {e}")
        return patterns

    def find_geometry_patterns(self, problem_card) -> List[Dict]:
        """
        Find what worked for similar geometries.
        Uses RAG semantic search when available; falls back to SQL LIKE.
        """
        # RAG path — semantic similarity over geometry + physics description
        if self.retriever and self.retriever.ready:
            try:
                results = self.retriever.find_similar_problems(
                    problem_card, top_k=5, success_only=True
                )
                if results:
                    logger.info(
                        f"RAG geometry patterns: {len(results)} similar problems found "
                        f"(top sim={results[0]['similarity']:.2f})"
                    )
                    return [
                        {
                            "model":      r.get("model", ""),
                            "r2":         r.get("r2", 0.0),
                            "similarity": r.get("similarity", 0.0),
                            "physics":    r.get("physics_type", ""),
                            "source":     "rag",
                        }
                        for r in results
                    ]
            except Exception as e:
                logger.warning(f"RAG geometry search failed, falling back to SQL: {e}")

        # SQL fallback
        geometry_desc = getattr(problem_card, "geometry_description", "") or ""
        results = []
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute("""
                    SELECT model_used, r2_score, notes
                    FROM runs
                    WHERE notes LIKE ? AND success = 1
                    ORDER BY r2_score DESC LIMIT 5
                """, (f"%{geometry_desc[:20]}%",)).fetchall()
                for row in rows:
                    results.append({
                        "model":  row[0],
                        "r2":     row[1],
                        "source": "sql",
                    })
        except Exception as e:
            logger.warning(f"Geometry pattern SQL search failed: {e}")
        return results
