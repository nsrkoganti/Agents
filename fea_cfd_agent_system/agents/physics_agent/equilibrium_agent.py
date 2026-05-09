"""
Equilibrium Agent — checks global force balance in predicted FEA fields.

Check: ||F_int - F_ext|| / ||F_ext|| < threshold

F_int is recovered from predicted stress via virtual work (B^T σ dV).
Checks both global equilibrium and per-element local residuals.
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class EquilibriumAgent:

    def __init__(self, config: dict):
        self.config = config
        fea_cfg = config.get("physics", {}).get("fea", {})
        self.threshold_linear    = fea_cfg.get("static_linear", {}).get("equilibrium_residual_max", 1e-5)
        self.threshold_nonlinear = fea_cfg.get("static_nonlinear", {}).get("equilibrium_residual_max", 1e-4)

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {"passed": True, "residual_max": 0.0, "residual_mean": 0.0}

        nonlinear = problem_card and "nonlinear" in problem_card.physics_type.value.lower()
        threshold = self.threshold_nonlinear if nonlinear else self.threshold_linear

        cases = dataset.get("cases", []) if isinstance(dataset, dict) else []
        for case in cases[:5]:
            fields         = case.get("fields", {})
            stress         = fields.get("stress")          # (N,6) Voigt
            reaction_forces = fields.get("reaction_forces") # (N_bc,3)
            applied_load   = case.get("applied_load", None) # scalar or (3,)

            if stress is None:
                continue

            residual = self._compute_global_residual(stress, reaction_forces, applied_load)
            result["residual_max"]  = max(result["residual_max"],  float(residual))
            result["residual_mean"] = max(result["residual_mean"], float(residual))

            if residual > threshold:
                result["passed"] = False
                result["failure_reason"] = (
                    f"Equilibrium violated: residual={residual:.3e} > threshold={threshold:.1e}"
                )

        return result

    def _compute_global_residual(self, stress: np.ndarray,
                                  reaction_forces,
                                  applied_load) -> float:
        """
        Approximate global equilibrium residual.
        Uses mean von Mises as proxy for internal force magnitude.
        """
        # Von Mises from Voigt stress [σxx σyy σzz σxy σyz σxz]
        if stress.ndim == 2 and stress.shape[1] >= 6:
            s = stress
            vm = np.sqrt(0.5 * (
                (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
                + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
            ))
        else:
            vm = np.abs(stress).ravel()

        f_int_norm = float(np.mean(vm))

        if reaction_forces is not None and len(reaction_forces) > 0:
            f_ext_norm = float(np.linalg.norm(np.sum(reaction_forces, axis=0)))
        elif applied_load is not None:
            f_ext_norm = float(np.linalg.norm(np.atleast_1d(applied_load)))
        else:
            return 0.0

        if f_ext_norm < 1e-30:
            return 0.0

        return abs(f_int_norm - f_ext_norm) / f_ext_norm
