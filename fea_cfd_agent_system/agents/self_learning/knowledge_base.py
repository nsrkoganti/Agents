"""
Knowledge Base — distills patterns from the run database.
Answers questions like:
  "What model should I try for this problem type?"
  "What lambda fixed this physics failure before?"
  "Has this exact problem been seen before?"
"""

import json
from typing import Optional, Dict, List
from loguru import logger
from agents.shared.llm_factory import get_dev_llm
from memory.run_database import RunDatabase


class KnowledgeBase:
    """
    The system's learned knowledge.
    Queries the run database and uses LLM to reason over patterns.
    """

    def __init__(self, config: dict, db: RunDatabase):
        self.config = config
        self.db     = db
        try:
            self.llm = get_dev_llm(max_tokens=1000)
        except Exception:
            self.llm = None

    def get_model_recommendation(self, state) -> Optional[Dict]:
        """Return best historical model for this problem type+mesh combination."""
        if state.problem_card is None:
            return None
        best = self.db.get_best_model_for(
            physics_type=state.problem_card.physics_type.value,
            mesh_type=state.problem_card.mesh_type.value,
        )
        return best

    def recommend_model_order(self, state) -> List[str]:
        """Based on past experience, what model order should we try?"""
        if state.problem_card is None:
            return []

        physics = state.problem_card.physics_type.value
        mesh    = state.problem_card.mesh_type.value

        best = self.db.get_best_model_for(physics, mesh)

        model_names = [
            "Transolver", "GINO", "MeshGraphNet", "FNO",
            "DeepONet", "PINN", "GPR", "XGBoost",
        ]
        rates = {}
        for m in model_names:
            rates[m] = self.db.get_success_rate(physics)

        sorted_models = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        ordered = [m for m, _ in sorted_models]

        if best and best.get("model_name") in ordered:
            ordered.remove(best["model_name"])
            ordered.insert(0, best["model_name"])

        logger.info(f"Knowledge-based model order: {ordered[:4]}")
        return ordered

    def recommend_lambda_weights(self, physics_type: str) -> Optional[Dict]:
        """What lambda weights have worked well for this physics type?"""
        weights = self.db.get_best_lambdas(physics_type)
        if weights:
            logger.info(f"Knowledge: using stored lambda weights for {physics_type}: {weights}")
        return weights

    def has_seen_similar_problem(self, state) -> Optional[Dict]:
        """Has the system seen a nearly identical problem before?"""
        if state.problem_card is None:
            return None
        import sqlite3
        try:
            pc = state.problem_card
            with sqlite3.connect(str(self.db.db_path)) as conn:
                row = conn.execute("""
                    SELECT model_used, r2_score, n_iterations
                    FROM runs
                    WHERE physics_type = ?
                      AND mesh_type = ?
                      AND ABS(data_size - ?) < 500
                      AND success = 1
                    ORDER BY r2_score DESC
                    LIMIT 1
                """, (
                    pc.physics_type.value,
                    pc.mesh_type.value,
                    pc.data_size,
                )).fetchone()

                if row:
                    return {
                        "model_name":   row[0],
                        "r2_score":     row[1],
                        "n_iterations": row[2],
                    }
        except Exception as e:
            logger.warning(f"Knowledge base query failed: {e}")
        return None

    def should_create_custom_model(self, state, attempts_so_far: int) -> bool:
        """Has the system failed enough to trigger custom model design?"""
        if attempts_so_far < 8:
            return False
        if state.problem_card is None:
            return False
        success_rate = self.db.get_success_rate(state.problem_card.physics_type.value)
        if success_rate < 0.4 and attempts_so_far >= 12:
            logger.warning(
                f"Low success rate ({success_rate:.0%}) on "
                f"{state.problem_card.physics_type.value} — triggering architect"
            )
            return True
        return False

    def get_system_stats(self) -> str:
        """Human-readable summary of what the system has learned."""
        stats = self.db.get_stats()
        return (
            f"System has completed {stats['total_runs']} runs "
            f"({stats['success_rate']:.0%} success rate), "
            f"created {stats['custom_models']} custom models, "
            f"discovered {stats['discovered_models']} new architectures."
        )
