"""Pattern recognizer — finds recurring success/failure patterns in run history."""

import sqlite3
import json
from typing import List, Dict
from loguru import logger
from memory.run_database import RunDatabase


class PatternRecognizer:
    """Identifies patterns in failures and fixes that led to success."""

    def __init__(self, db: RunDatabase):
        self.db = db

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

    def find_geometry_patterns(self, geometry_desc: str) -> List[Dict]:
        """Find what worked for similar geometries."""
        results = []
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute("""
                    SELECT winning_model, r2_score, lambda_weights
                    FROM runs
                    WHERE geometry_description LIKE ? AND success = 1
                    ORDER BY r2_score DESC LIMIT 5
                """, (f"%{geometry_desc[:20]}%",)).fetchall()
                for row in rows:
                    results.append({
                        "model":          row[0],
                        "r2":             row[1],
                        "lambda_weights": json.loads(row[2]) if row[2] else {},
                    })
        except Exception as e:
            logger.warning(f"Geometry pattern search failed: {e}")
        return results
