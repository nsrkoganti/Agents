"""
Turbulence Physics Agent.
Checks: regime (Re), k-omega SST consistency, y+ validity,
turbulence intensity bounds.
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class TurbulenceAgent:

    def __init__(self, config: dict):
        self.config = config

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        if problem_card is None or "CFD" not in problem_card.physics_type.value:
            return {"passed": True, "reason": "N/A for non-CFD physics"}

        result = {
            "passed":       True,
            "re_number":    problem_card.re_number,
            "regime":       "unknown",
            "k_positive":   True,
            "omega_positive": True,
            "y_plus_ok":    True,
            "intensity_ok": True,
        }

        Re = problem_card.re_number or 0
        if Re < 2300:
            result["regime"] = "laminar"
        elif Re < 4000:
            result["regime"] = "transitional"
        else:
            result["regime"] = "turbulent"

        for case in dataset.get("cases", [])[:5]:
            fields  = case.get("fields", {})

            k_field = fields.get("k") or fields.get("tke")
            if k_field is not None:
                if np.any(k_field < 0):
                    result["k_positive"]   = False
                    result["passed"]       = False
                    result["failure_reason"] = f"Negative TKE (k<0): min={k_field.min():.3e}"

            omega_field = fields.get("omega")
            if omega_field is not None:
                if np.any(omega_field < 0):
                    result["omega_positive"] = False
                    result["passed"]         = False
                    result["failure_reason"] = f"Negative omega: min={omega_field.min():.3e}"

            yplus = fields.get("yPlus") or fields.get("y_plus")
            if yplus is not None:
                tm = problem_card.turbulence_model or ""
                if "LowRe" in tm or "low_re" in tm:
                    if float(np.max(yplus)) > 1.0:
                        result["y_plus_ok"]    = False
                        result["passed"]       = False
                        result["failure_reason"] = f"y+ too high for Low-Re: max={np.max(yplus):.2f}"

            velocity = fields.get("velocity")
            if k_field is not None and velocity is not None:
                U_mag = np.linalg.norm(velocity, axis=-1) if velocity.ndim > 1 else velocity
                U_ref = float(np.mean(U_mag)) + 1e-10
                I     = np.sqrt(2.0 * k_field / 3.0) / U_ref
                I_mean = float(np.mean(I))
                if I_mean > 0.50:
                    result["intensity_ok"]   = False
                    result["passed"]         = False
                    result["failure_reason"] = f"Turbulence intensity too high: I={I_mean:.1%}"
                elif I_mean < 0.001:
                    result["intensity_ok"]   = False
                    result["passed"]         = False
                    result["failure_reason"] = f"Turbulence intensity too low: I={I_mean:.4%}"
                result["intensity_mean"] = round(I_mean, 4)

        return result
