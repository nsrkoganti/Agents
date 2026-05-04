"""
Experience Memory — thin wrapper over RunDatabase providing agent-friendly API.
Stores all run history and provides structured retrieval.
"""

from memory.run_database import RunDatabase, RunRecord, FailureRecord
from typing import Optional, Dict, List


class ExperienceMemory:
    """Agent-facing interface to the persistent run database."""

    def __init__(self, db: RunDatabase):
        self.db = db

    def remember_run(self, record: RunRecord):
        self.db.save_run(record)

    def remember_failure(self, record: FailureRecord):
        self.db.save_failure(record)

    def recall_best_model(self, physics_type: str, mesh_type: str, n_data: int) -> Optional[str]:
        return self.db.get_best_model_for_problem(physics_type, mesh_type, n_data)

    def recall_best_lambdas(self, failure_type: str, physics_type: str) -> Optional[Dict]:
        return self.db.get_best_lambda_for_failure(failure_type, physics_type)

    def success_rate(self, model_name: str, physics_type: str) -> float:
        return self.db.get_success_rate(model_name, physics_type)

    def stats(self) -> Dict:
        return self.db.stats()
