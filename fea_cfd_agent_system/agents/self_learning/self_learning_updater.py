"""
Self-Learning Updater — runs after every completed pipeline run.
Saves results to database. Updates knowledge patterns.
"""

import uuid
import datetime
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus
from memory.run_database import RunDatabase, RunRecord, FailureRecord


class SelfLearningUpdater:
    """
    Updates the system's knowledge after every run.
    This is what makes the system get smarter over time.
    """

    def __init__(self, config: dict, db: RunDatabase):
        self.config = config
        self.db     = db

    def update(self, state: AgentSystemState) -> AgentSystemState:
        """Record everything from this run into permanent memory."""
        try:
            pc = state.problem_card
            ev = state.evaluation_result or state.training_result

            run_id = state.run_id or str(uuid.uuid4())
            ts     = datetime.datetime.utcnow().isoformat()

            r2     = getattr(ev, "r2_score",    0.0) or 0.0
            rel_l2 = getattr(ev, "rel_l2_error", 1.0) or 1.0
            success    = state.pipeline_success
            model_name = (
                state.selected_model.name if state.selected_model else "unknown"
            )

            record = RunRecord(
                run_id       = run_id,
                physics_type = pc.physics_type.value if pc else "unknown",
                mesh_type    = pc.mesh_type.value    if pc else "unknown",
                data_size    = pc.data_size          if pc else 0,
                model_used   = model_name,
                r2_score     = r2,
                rel_l2       = rel_l2,
                success      = success,
                n_iterations = state.current_attempt,
                timestamp    = ts,
            )
            self.db.save_run(record)

            for rec in (state.iteration_log or []):
                if rec.failure_reason:
                    fail_r2 = 0.0
                    if rec.evaluation_result:
                        fail_r2 = rec.evaluation_result.r2_score or 0.0
                    self.db.save_failure(FailureRecord(
                        run_id        = run_id,
                        model_name    = rec.model_name,
                        failure_reason= rec.failure_reason or "unknown",
                        fix_tried     = rec.fix_applied   or "none",
                        r2_at_failure = fail_r2,
                        iteration     = rec.attempt_number,
                        physics_type  = pc.physics_type.value if pc else "unknown",
                        timestamp     = ts,
                    ))

            if state.selected_model and pc:
                self.db.save_model_performance(
                    model_name   = model_name,
                    physics_type = pc.physics_type.value,
                    mesh_type    = pc.mesh_type.value,
                    r2           = r2,
                    rel_l2       = rel_l2,
                    success      = success,
                    data_size    = pc.data_size,
                )

            stats = self.db.get_stats()
            logger.info(
                f"Self-learning updated: "
                f"{stats['total_runs']} total runs, "
                f"{stats['success_rate']:.0%} success rate, "
                f"{stats['custom_models']} custom models"
            )

        except Exception as e:
            logger.warning(f"Self-learning update failed: {e}")

        return state
