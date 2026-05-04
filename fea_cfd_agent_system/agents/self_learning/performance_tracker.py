"""Performance tracker — aggregates model performance statistics across runs."""

from typing import Dict, List
from memory.run_database import RunDatabase


class PerformanceTracker:
    """Tracks model performance over time and computes ranking statistics."""

    def __init__(self, db: RunDatabase):
        self.db = db

    def get_model_ranking(self, physics_type: str) -> List[Dict]:
        """Return models ranked by historical success rate for a physics type."""
        import sqlite3
        rankings = []
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute("""
                    SELECT model_name,
                           AVG(r2_achieved) as avg_r2,
                           SUM(success) * 1.0 / COUNT(*) as success_rate,
                           COUNT(*) as total_runs
                    FROM model_performance
                    WHERE physics_type = ?
                    GROUP BY model_name
                    ORDER BY avg_r2 DESC
                """, (physics_type,)).fetchall()
                for row in rows:
                    rankings.append({
                        "model":        row[0],
                        "avg_r2":       row[1],
                        "success_rate": row[2],
                        "total_runs":   row[3],
                    })
        except Exception:
            pass
        return rankings

    def get_trend(self, model_name: str, physics_type: str, last_n: int = 10) -> List[float]:
        """Return R2 trend for a model over the last N runs."""
        import sqlite3
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute("""
                    SELECT r2_achieved FROM model_performance
                    WHERE model_name = ? AND physics_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (model_name, physics_type, last_n)).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []
