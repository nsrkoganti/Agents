"""
Evaluator Agent — computes accuracy metrics and diagnoses failures.
"""

import time
import numpy as np
import torch
from loguru import logger
from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, EvaluationResult
)


class EvaluatorAgent:

    def __init__(self, config: dict):
        self.config = config
        thresholds = config.get("evaluation", {}).get("thresholds", {})
        self.r2_min       = thresholds.get("r2_min", 0.92)
        self.rel_l2_max   = thresholds.get("rel_l2_max", 0.05)
        self.max_pt_max   = thresholds.get("max_point_error_max", 0.15)
        self.infer_ms_max = thresholds.get("inference_time_ms_max", 100.0)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Evaluator Agent: computing metrics")
        state.evaluator_status = AgentStatus.RUNNING

        if state.training_result is None or state.training_result.model_object is None:
            result = EvaluationResult(passed=False, failure_reason="No model produced by trainer")
            state.evaluation_result = result
            state.evaluator_status = AgentStatus.FAILED
            return state

        model   = state.training_result.model_object
        dataset = state.dataset
        result  = EvaluationResult()
        elapsed_ms = 0.0

        try:
            cases  = dataset.get("cases", [])
            n_test = max(1, int(len(cases) * 0.2))
            test_cases = cases[-n_test:]

            preds, targets = [], []
            for case in test_cases:
                target = self._extract_target(case, state.problem_card)
                if target is None:
                    continue
                start = time.perf_counter()
                pred  = self._run_inference(model, case)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if pred is not None:
                    preds.append(pred.flatten())
                    targets.append(target.flatten())

            if not preds:
                result.passed = False
                result.failure_reason = "No valid predictions produced"
                state.evaluation_result = result
                state.evaluator_status = AgentStatus.FAILED
                return state

            preds   = np.concatenate(preds)
            targets = np.concatenate(targets)

            result.r2_score        = self._r2(targets, preds)
            result.rel_l2_error    = self._rel_l2(targets, preds)
            result.max_point_error = float(np.max(np.abs(targets - preds)) /
                                           (np.max(np.abs(targets)) + 1e-10))
            result.inference_time_ms = elapsed_ms

            passed = (
                result.r2_score        >= self.r2_min     and
                result.rel_l2_error    <= self.rel_l2_max and
                result.max_point_error <= self.max_pt_max and
                result.inference_time_ms <= self.infer_ms_max
            )
            result.passed = passed

            if not passed:
                result.failure_reason  = self._diagnose_failure(result)
                result.recommended_fix = self._recommend_fix(result)

            logger.info(
                f"Evaluation: R2={result.r2_score:.4f}, "
                f"L2={result.rel_l2_error:.4f}, "
                f"passed={result.passed}"
            )

        except Exception as e:
            result.passed = False
            result.failure_reason = str(e)
            logger.error(f"Evaluator error: {e}")

        state.evaluation_result = result
        state.evaluator_status = AgentStatus.PASSED if result.passed else AgentStatus.FAILED
        return state

    def _r2(self, y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def _rel_l2(self, y_true, y_pred) -> float:
        return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-10))

    def _extract_target(self, case, problem_card):
        fields = case.get("fields", {})
        if problem_card and problem_card.output_targets:
            for target in problem_card.output_targets:
                for key in [target, target.replace("_field", ""), target.split("_")[0]]:
                    if key in fields:
                        return fields[key]
        if fields:
            return list(fields.values())[0]
        return None

    def _run_inference(self, model, case):
        try:
            if model is None:
                return None
            with torch.no_grad():
                x = self._case_to_tensor(case)
                if x is None:
                    return None
                out = model(x)
                return out.cpu().numpy() if hasattr(out, "cpu") else np.array(out)
        except Exception:
            return None

    def _case_to_tensor(self, case):
        coords = case.get("node_coords")
        if coords is None:
            return None
        return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

    def _diagnose_failure(self, result: EvaluationResult) -> str:
        if result.r2_score < 0.70:
            return "Severe underfitting — model too simple or wrong type for this physics"
        if result.r2_score > 0.95 and result.rel_l2_error > self.rel_l2_max:
            return "High R2 but L2 error fails — large local errors somewhere in domain"
        if result.r2_score < self.r2_min:
            return f"Underfitting — R2={result.r2_score:.3f} below threshold {self.r2_min}"
        if result.rel_l2_error > self.rel_l2_max:
            return f"L2 error too high: {result.rel_l2_error:.4f} > {self.rel_l2_max}"
        if result.inference_time_ms > self.infer_ms_max:
            return f"Too slow: {result.inference_time_ms:.1f}ms > {self.infer_ms_max}ms"
        return "Metrics below threshold"

    def _recommend_fix(self, result: EvaluationResult) -> str:
        if result.r2_score < 0.70:
            return "next_model"
        if result.r2_score < self.r2_min:
            return "tune_hyperparameters"
        if result.rel_l2_error > self.rel_l2_max * 2:
            return "increase_physics_loss"
        return "tune_hyperparameters"
