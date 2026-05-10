"""
Persistent SQLite database for all agent runs, failures, and learned patterns.
Central memory store imported by knowledge_base, self_learning_updater, and architect_agent.
"""

import sqlite3
import json
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class RunRecord:
    run_id: str
    physics_type: str
    mesh_type: str
    data_size: int
    model_used: str
    r2_score: float
    rel_l2: float
    success: bool
    n_iterations: int
    timestamp: str
    problem_hash: str = ""
    notes: str = ""


@dataclass
class FailureRecord:
    run_id: str
    model_name: str
    failure_reason: str
    fix_tried: str
    r2_at_failure: float
    iteration: int
    physics_type: str
    timestamp: str


class RunDatabase:
    """
    SQLite-backed persistent memory for the agent system.
    Stores runs, failures, model performance, patterns, and custom models.
    Optionally indexes every save into the RAG vector store when a retriever is injected.
    """

    def __init__(self, db_path: str = "memory/experience.db", retriever=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retriever = retriever  # Optional[RAGRetriever]
        self._init_db()
        logger.info(f"RunDatabase initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create all tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    physics_type TEXT,
                    mesh_type    TEXT,
                    data_size    INTEGER,
                    model_used   TEXT,
                    r2_score     REAL,
                    rel_l2       REAL,
                    success      INTEGER,
                    n_iterations INTEGER,
                    problem_hash TEXT,
                    notes        TEXT,
                    timestamp    TEXT
                );

                CREATE TABLE IF NOT EXISTS failures (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id         TEXT,
                    model_name     TEXT,
                    failure_reason TEXT,
                    fix_tried      TEXT,
                    r2_at_failure  REAL,
                    iteration      INTEGER,
                    physics_type   TEXT,
                    timestamp      TEXT
                );

                CREATE TABLE IF NOT EXISTS model_performance (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name   TEXT,
                    physics_type TEXT,
                    mesh_type    TEXT,
                    r2_score     REAL,
                    rel_l2       REAL,
                    success      INTEGER,
                    data_size    INTEGER,
                    timestamp    TEXT
                );

                CREATE TABLE IF NOT EXISTS physics_patterns (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    physics_type TEXT,
                    mesh_type    TEXT,
                    best_model   TEXT,
                    avg_r2       REAL,
                    success_rate REAL,
                    lambda_json  TEXT,
                    n_samples    INTEGER,
                    updated_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS discovered_models (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    name          TEXT UNIQUE,
                    arxiv_id      TEXT,
                    mesh_type     TEXT,
                    physics_types TEXT,
                    benchmark_l2  REAL,
                    github_url    TEXT,
                    discovered_at TEXT
                );

                CREATE TABLE IF NOT EXISTS custom_models (
                    model_id    TEXT PRIMARY KEY,
                    name        TEXT,
                    dna_json    TEXT,
                    code        TEXT,
                    physics_type TEXT,
                    r2_score    REAL,
                    generation  INTEGER,
                    created_at  TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runs_physics    ON runs (physics_type);
                CREATE INDEX IF NOT EXISTS idx_runs_success    ON runs (success);
                CREATE INDEX IF NOT EXISTS idx_failures_model  ON failures (model_name);
                CREATE INDEX IF NOT EXISTS idx_model_perf      ON model_performance (model_name, physics_type);
            """)

    # ──────────────── Run records ────────────────

    def save_run(self, record: RunRecord):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs
                (run_id, physics_type, mesh_type, data_size, model_used,
                 r2_score, rel_l2, success, n_iterations, problem_hash, notes, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record.run_id, record.physics_type, record.mesh_type,
                record.data_size, record.model_used, record.r2_score,
                record.rel_l2, int(record.success), record.n_iterations,
                record.problem_hash, record.notes, record.timestamp,
            ))
        if self.retriever:
            self.retriever.index_run(record)

    def save_failure(self, record: FailureRecord, r2_after: float = 0.0,
                     failed_checks: Optional[List[str]] = None,
                     lambda_json: Optional[Dict] = None):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO failures
                (run_id, model_name, failure_reason, fix_tried,
                 r2_at_failure, iteration, physics_type, timestamp)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                record.run_id, record.model_name, record.failure_reason,
                record.fix_tried, record.r2_at_failure, record.iteration,
                record.physics_type, record.timestamp,
            ))
        if self.retriever:
            self.retriever.index_failure(record, r2_after, failed_checks, lambda_json)

    def save_model_performance(self, model_name: str, physics_type: str,
                                mesh_type: str, r2: float, rel_l2: float,
                                success: bool, data_size: int):
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO model_performance
                (model_name, physics_type, mesh_type, r2_score, rel_l2,
                 success, data_size, timestamp)
                VALUES (?,?,?,?,?,?,?,?)
            """, (model_name, physics_type, mesh_type, r2, rel_l2,
                  int(success), data_size, ts))

    # ──────────────── Query helpers ────────────────

    def get_best_model_for(self, physics_type: str,
                            mesh_type: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT model_name, AVG(r2_score) as avg_r2,
                       SUM(success) * 1.0 / COUNT(*) as success_rate,
                       COUNT(*) as n
                FROM model_performance
                WHERE physics_type = ? AND mesh_type = ?
                GROUP BY model_name
                HAVING n >= 2
                ORDER BY avg_r2 DESC
                LIMIT 1
            """, (physics_type, mesh_type)).fetchone()
            return dict(row) if row else None

    def get_failure_patterns(self, physics_type: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT model_name, failure_reason, COUNT(*) as count
                FROM failures
                WHERE physics_type = ?
                GROUP BY model_name, failure_reason
                ORDER BY count DESC
                LIMIT 10
            """, (physics_type,)).fetchall()
            return [dict(r) for r in rows]

    def get_recent_runs(self, n: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?
            """, (n,)).fetchall()
            return [dict(r) for r in rows]

    def get_success_rate(self, physics_type: str) -> float:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT AVG(success) as rate FROM runs WHERE physics_type = ?
            """, (physics_type,)).fetchone()
            return float(row["rate"]) if row and row["rate"] is not None else 0.5

    def get_best_lambdas(self, physics_type: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT lambda_json FROM physics_patterns
                WHERE physics_type = ?
                ORDER BY avg_r2 DESC LIMIT 1
            """, (physics_type,)).fetchone()
            if row and row["lambda_json"]:
                return json.loads(row["lambda_json"])
            return None

    def save_physics_pattern(self, physics_type: str, mesh_type: str,
                              best_model: str, avg_r2: float,
                              success_rate: float, lambdas: dict,
                              n_samples: int):
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO physics_patterns
                (physics_type, mesh_type, best_model, avg_r2,
                 success_rate, lambda_json, n_samples, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (physics_type, mesh_type, best_model, avg_r2,
                  success_rate, json.dumps(lambdas), n_samples, ts))

    # ──────────────── Discovered models ────────────────

    def save_discovered_model(self, name: str, arxiv_id: str,
                               mesh_type: str, physics_types: list,
                               benchmark_l2: float, github_url: str = ""):
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO discovered_models
                (name, arxiv_id, mesh_type, physics_types, benchmark_l2,
                 github_url, discovered_at)
                VALUES (?,?,?,?,?,?,?)
            """, (name, arxiv_id, mesh_type, json.dumps(physics_types),
                  benchmark_l2, github_url, ts))

    def get_discovered_models(self, physics_type: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM discovered_models
                ORDER BY benchmark_l2 ASC LIMIT 20
            """).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                types = json.loads(d.get("physics_types", "[]"))
                if not physics_type or physics_type in types or not types:
                    result.append(d)
            return result

    # ──────────────── Custom models ────────────────

    def save_custom_model(self, model_id: str, name: str, dna: dict,
                           code: str, problem: str, r2: float,
                           generation: int,
                           failed_checks: Optional[List[str]] = None):
        ts = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO custom_models
                (model_id, name, dna_json, code, physics_type,
                 r2_score, generation, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (model_id, name, json.dumps(dna), code, problem,
                  r2, generation, ts))
        if self.retriever:
            self.retriever.index_custom_model(
                model_id, name, dna, code, problem, r2, failed_checks or []
            )

    def get_best_custom_model(self, physics_type: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM custom_models
                WHERE physics_type = ?
                ORDER BY r2_score DESC LIMIT 1
            """, (physics_type,)).fetchone()
            return dict(row) if row else None

    def update_custom_model_r2(self, model_id: str, r2: float):
        with self._connect() as conn:
            conn.execute("""
                UPDATE custom_models SET r2_score = ? WHERE model_id = ?
            """, (r2, model_id))

    # ──────────────── Statistics ────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            n_runs    = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            n_success = conn.execute("SELECT COUNT(*) FROM runs WHERE success=1").fetchone()[0]
            n_custom  = conn.execute("SELECT COUNT(*) FROM custom_models").fetchone()[0]
            n_disc    = conn.execute("SELECT COUNT(*) FROM discovered_models").fetchone()[0]
            return {
                "total_runs":        n_runs,
                "successful_runs":   n_success,
                "success_rate":      n_success / max(n_runs, 1),
                "custom_models":     n_custom,
                "discovered_models": n_disc,
            }
