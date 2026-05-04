"""
Tracks the family tree of custom-designed models.
Records which DNA templates were used, mutations, and performance across generations.
"""

import json
import datetime
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger


class ModelGenealogy:
    """
    Tracks lineage of architect-designed models.
    Helps understand which mutations improve performance.
    """

    def __init__(self, db_path: str = "memory/experience.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_genealogy (
                    model_id    TEXT PRIMARY KEY,
                    parent_id   TEXT,
                    name        TEXT,
                    family      TEXT,
                    generation  INTEGER,
                    dna_diff    TEXT,
                    r2_score    REAL,
                    physics_type TEXT,
                    created_at  TEXT
                )
            """)

    def record_model(self, model_id: str, name: str, family: str,
                     generation: int, dna: dict, r2: float,
                     physics_type: str, parent_id: Optional[str] = None):
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_genealogy
                (model_id, parent_id, name, family, generation,
                 dna_diff, r2_score, physics_type, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (model_id, parent_id, name, family, generation,
                  json.dumps(dna), r2, physics_type, ts))

    def get_best_ancestor(self, physics_type: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM model_genealogy
                WHERE physics_type = ?
                ORDER BY r2_score DESC LIMIT 1
            """, (physics_type,)).fetchone()
            return dict(row) if row else None

    def get_lineage(self, model_id: str) -> List[Dict]:
        """Walk up the family tree."""
        lineage = []
        current_id = model_id
        with self._connect() as conn:
            while current_id:
                row = conn.execute("""
                    SELECT * FROM model_genealogy WHERE model_id = ?
                """, (current_id,)).fetchone()
                if not row:
                    break
                lineage.append(dict(row))
                current_id = row["parent_id"]
        return lineage

    def get_generation_stats(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT generation, AVG(r2_score) as avg_r2, COUNT(*) as n
                FROM model_genealogy
                GROUP BY generation
                ORDER BY generation
            """).fetchall()
            return [dict(r) for r in rows]
