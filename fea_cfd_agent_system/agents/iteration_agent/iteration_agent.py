"""
Iteration Agent — root-cause analysis of failures.
Decides: fix same model, retry with new config, or move to next model.
"""

import json
import datetime
from loguru import logger
from agents.shared.llm_factory import get_dev_llm
from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, IterationRecord
)


class IterationAgent:
    """
    Analyzes failures and decides what to fix.
    Uses LLM to reason about the best fix strategy.
    """

    def __init__(self, config: dict):
        self.config      = config
        self.llm         = get_dev_llm(max_tokens=1000)
        self.max_attempts  = config.get("iteration", {}).get("total_max_attempts", 24)
        self.max_per_model = config.get("iteration", {}).get("max_attempts_per_model", 3)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        state.current_attempt += 1
        logger.info(f"Iteration Agent: attempt {state.current_attempt}/{self.max_attempts}")

        if state.current_attempt >= self.max_attempts:
            state.iteration_status = AgentStatus.FAILED
            state.error_message    = f"Max {self.max_attempts} attempts reached without success"
            return state

        fix_plan = self._diagnose_and_plan(state)

        record = IterationRecord(
            attempt_number=state.current_attempt,
            model_name=state.selected_model.name if state.selected_model else "unknown",
            evaluation_result=state.evaluation_result,
            physics_report=state.physics_report,
            overall_passed=False,
            failure_reason=self._get_failure_reason(state),
            fix_applied=fix_plan,
        )
        state.iteration_log.append(record)

        state = self._apply_fix(state, fix_plan)
        state.iteration_status = AgentStatus.RUNNING

        self._maybe_request_more_data(state)

        logger.info(f"Fix applied: {fix_plan[:120]}")
        return state

    def _maybe_request_more_data(self, state: AgentSystemState) -> None:
        """Publish REQUEST_MORE_DATA if R2 is persistently low after 6 attempts."""
        if state.current_attempt < 6:
            return
        r2 = state.evaluation_result.r2_score if state.evaluation_result else 0.0
        if r2 >= 0.5:
            return
        # Only request once — check existing unhandled messages
        already_requested = any(
            m.get("type") == "REQUEST_MORE_DATA" and not m.get("handled")
            for m in state.agent_messages
        )
        if already_requested:
            return
        data_size = state.problem_card.data_size if state.problem_card else 0
        state.agent_messages.append({
            "from":      "iteration_agent",
            "to":        "dataset_agent",
            "type":      "REQUEST_MORE_DATA",
            "reason":    (f"R2={r2:.2f} after {state.current_attempt} attempts — "
                          f"need larger/better dataset (current size={data_size})"),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "handled":   False,
        })
        logger.info(f"Published REQUEST_MORE_DATA (R2={r2:.2f} after {state.current_attempt} attempts)")

    def _diagnose_and_plan(self, state: AgentSystemState) -> str:
        eval_r = state.evaluation_result
        phys_r = state.physics_report

        diagnosis_info = {
            "attempt": state.current_attempt,
            "model":   state.selected_model.name if state.selected_model else "unknown",
            "evaluator": {
                "r2":     eval_r.r2_score if eval_r else None,
                "rel_l2": eval_r.rel_l2_error if eval_r else None,
                "passed": eval_r.passed if eval_r else False,
                "reason": eval_r.failure_reason if eval_r else None,
            },
            "physics": {
                "overall_passed": phys_r.overall_passed if phys_r else False,
                "governing_eq":   phys_r.governing_equations_passed if phys_r else True,
                "bc":             phys_r.boundary_conditions_passed if phys_r else True,
                "conservation":   phys_r.conservation_passed if phys_r else True,
                "turbulence":     phys_r.turbulence_passed if phys_r else True,
                "fix_instructions": phys_r.fix_instructions if phys_r else None,
            },
            "attempts_on_current_model": self._count_model_attempts(state),
        }

        prompt = f"""
You are debugging a failed ML surrogate model for CFD/FEA simulation.

Failure info:
{json.dumps(diagnosis_info, indent=2)}

Decide the fix. Output ONLY JSON:
{{
  "fix_type": "one of: tune_hyperparameters | increase_physics_loss | re_encode_bc | next_model | add_data_augmentation | switch_to_pinn",
  "fix_description": "what to change and why",
  "lambda_updates": {{"continuity": 3.0, "bc": 5.0}},
  "next_model": null
}}

Rules:
- If R2 < 0.7 and first attempt: fix_type = next_model
- If R2 between 0.7 and threshold: fix_type = tune_hyperparameters
- If physics BC failed: fix_type = increase_physics_loss with lambda_updates.bc *= 5
- If continuity failed: fix_type = increase_physics_loss with lambda_updates.continuity *= 3
- If attempts_on_current_model >= 3: fix_type = next_model
- If all physics fail repeatedly: fix_type = switch_to_pinn
"""
        try:
            response = self.llm.invoke(prompt)
            content  = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return content
        except Exception as e:
            logger.warning(f"LLM diagnosis failed: {e}")
            return json.dumps({"fix_type": "next_model", "fix_description": "LLM failed, trying next model"})

    def _apply_fix(self, state: AgentSystemState, fix_plan: str) -> AgentSystemState:
        try:
            plan     = json.loads(fix_plan)
            fix_type = plan.get("fix_type", "next_model")

            if fix_type == "next_model":
                state.current_model_index += 1
                if state.current_model_index < len(state.ranked_shortlist):
                    state.selected_model = state.ranked_shortlist[state.current_model_index]
                    logger.info(f"Switching to: {state.selected_model.name}")
                else:
                    state.error_message = "All models in shortlist exhausted"

            elif fix_type == "increase_physics_loss":
                updates = plan.get("lambda_updates", {})
                for k, v in updates.items():
                    state.physics_lambda_weights[k] = v
                logger.info(f"Updated physics lambdas: {updates}")

            elif fix_type == "tune_hyperparameters":
                state.unified_schema["use_optuna"] = True

            elif fix_type == "switch_to_pinn":
                for m in state.all_candidates:
                    if m.name == "PINN":
                        state.selected_model = m
                        break

        except json.JSONDecodeError:
            state.current_model_index += 1

        return state

    def _count_model_attempts(self, state: AgentSystemState) -> int:
        if not state.selected_model:
            return 0
        return sum(
            1 for r in state.iteration_log
            if r.model_name == state.selected_model.name
        )

    def _get_failure_reason(self, state: AgentSystemState) -> str:
        reasons = []
        if state.evaluation_result and not state.evaluation_result.passed:
            reasons.append(f"Accuracy: R2={state.evaluation_result.r2_score:.3f}")
        if state.physics_report and not state.physics_report.overall_passed:
            reasons.append(f"Physics: {state.physics_report.fix_instructions}")
        return " | ".join(reasons) if reasons else "Unknown"
