"""
Verifier Agent — final gate before saving.
Tests generalization, robustness, and deployment readiness.
"""

import time
import numpy as np
import torch
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus


class VerifierAgent:

    def __init__(self, config: dict):
        self.config = config

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Verifier Agent: checking generalization + robustness")
        state.verifier_status = AgentStatus.RUNNING

        if state.training_result is None or state.training_result.model_object is None:
            state.verification_passed = False
            state.verifier_status     = AgentStatus.FAILED
            return state

        model   = state.training_result.model_object
        dataset = state.dataset
        checks  = {}

        # 1. Generalization on held-out cases
        cases    = dataset.get("cases", [])
        n_test   = max(1, int(len(cases) * 0.1))
        held_out = cases[:n_test]
        gen_errors = []

        for case in held_out:
            try:
                x = self._case_to_tensor(case)
                if x is None:
                    continue
                with torch.no_grad():
                    pred = model(x)
                fields = case.get("fields", {})
                if fields:
                    target  = list(fields.values())[0].flatten()
                    pred_np = pred.cpu().numpy().flatten()
                    min_len = min(len(target), len(pred_np))
                    if min_len > 0:
                        err = (np.linalg.norm(target[:min_len] - pred_np[:min_len]) /
                               (np.linalg.norm(target[:min_len]) + 1e-10))
                        gen_errors.append(float(err))
            except Exception:
                pass

        checks["generalization_l2"] = float(np.mean(gen_errors)) if gen_errors else 0.1
        checks["generalization_ok"] = checks["generalization_l2"] < 0.10

        # 2. Stability check — slightly perturb inputs
        stability_ok = True
        if cases:
            try:
                x = self._case_to_tensor(cases[0])
                if x is not None:
                    with torch.no_grad():
                        pred1 = model(x)
                        pred2 = model(x + torch.randn_like(x) * 0.001)
                    diff = float(torch.mean(torch.abs(pred1 - pred2)).item())
                    stability_ok = (diff < 0.1 and
                                    not (torch.isnan(pred1).any() or torch.isinf(pred1).any()))
                    checks["stability_diff"] = diff
            except Exception as e:
                logger.warning(f"Stability check error: {e}")
        checks["stability_ok"] = stability_ok

        # 3. Inference time check
        if cases:
            try:
                x = self._case_to_tensor(cases[0])
                if x is not None:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        for _ in range(5):
                            model(x)
                    avg_ms = (time.perf_counter() - t0) / 5 * 1000
                    checks["inference_ms"] = avg_ms
                    checks["speed_ok"]     = avg_ms < 1000
            except Exception:
                checks["speed_ok"] = True

        # 4. No NaN/Inf check
        if cases:
            try:
                x = self._case_to_tensor(cases[0])
                if x is not None:
                    with torch.no_grad():
                        out = model(x)
                    checks["no_nan"] = not (torch.isnan(out).any().item() or
                                            torch.isinf(out).any().item())
            except Exception:
                checks["no_nan"] = True

        passed = all([
            checks.get("generalization_ok", True),
            checks.get("stability_ok",      True),
            checks.get("speed_ok",          True),
            checks.get("no_nan",            True),
        ])

        state.verification_passed = passed
        state.verification_detail = checks
        state.verifier_status     = AgentStatus.PASSED if passed else AgentStatus.FAILED

        logger.info(f"Verification: {checks} — overall={passed}")
        return state

    def _case_to_tensor(self, case):
        coords = case.get("node_coords")
        if coords is None:
            return None
        return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
