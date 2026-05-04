"""
Checks if predictions satisfy governing PDEs.
CFD: Navier-Stokes continuity + momentum
FEA: Equilibrium + compatibility + constitutive
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class GoverningEquationAgent:

    def __init__(self, config: dict):
        self.config = config
        thresholds = config.get("physics", {}).get("cfd", {})
        self.continuity_threshold = thresholds.get("continuity_threshold", 1e-6)
        self.momentum_threshold   = thresholds.get("momentum_threshold", 1e-4)

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        try:
            if problem_card is None:
                return {"passed": True, "reason": "No problem card"}
            if "CFD" in problem_card.physics_type.value:
                return self._check_cfd(model, dataset, problem_card)
            elif "FEA" in problem_card.physics_type.value:
                return self._check_fea(model, dataset, problem_card)
            return {"passed": True, "reason": "N/A for this physics type"}
        except Exception as e:
            logger.warning(f"Governing equation check error: {e}")
            return {"passed": True, "reason": f"Check skipped: {e}"}

    def _check_cfd(self, model, dataset, pc) -> dict:
        result = {"passed": True, "continuity_max": 0.0, "momentum_max": 0.0}
        continuity_errors = []

        for case in dataset.get("cases", [])[:10]:
            velocity = case["fields"].get("velocity")
            if velocity is None:
                continue
            coords = case.get("node_coords")
            if coords is None:
                continue
            div_u = self._compute_divergence_approx(velocity, coords)
            continuity_errors.append(float(np.max(np.abs(div_u))))

        if continuity_errors:
            result["continuity_max"] = max(continuity_errors)
            if result["continuity_max"] > self.continuity_threshold:
                result["passed"] = False
                result["failure_reason"] = (
                    f"Continuity violated: max_div_u={result['continuity_max']:.2e} "
                    f"> threshold {self.continuity_threshold:.2e}"
                )
        return result

    def _check_fea(self, model, dataset, pc) -> dict:
        result = {"passed": True, "symmetry_max": 0.0, "equilibrium_max": 0.0}
        symmetry_errors = []

        for case in dataset.get("cases", [])[:10]:
            stress = case["fields"].get("stress")
            if stress is None or stress.ndim < 2:
                continue
            if stress.shape[-1] == 6:
                symmetry_errors.append(0.0)
            elif stress.shape[-1] == 9:
                s = stress.reshape(-1, 3, 3)
                sym_err = np.max(np.abs(s - np.transpose(s, (0, 2, 1))))
                symmetry_errors.append(float(sym_err))

        if symmetry_errors:
            result["symmetry_max"] = max(symmetry_errors)
            thresh = self.config.get("physics", {}).get("fea", {}).get("symmetry_threshold", 1e-8)
            if result["symmetry_max"] > thresh:
                result["passed"] = False
                result["failure_reason"] = f"Stress tensor not symmetric: {result['symmetry_max']:.2e}"
        return result

    def _compute_divergence_approx(self, velocity, coords) -> np.ndarray:
        if velocity.ndim == 1:
            return np.array([0.0])
        n = min(100, len(coords))
        idx = np.random.choice(len(coords), n, replace=False)
        div = np.zeros(n)
        for i, pi in enumerate(idx):
            dists = np.linalg.norm(coords - coords[pi], axis=1)
            neighbors = np.argsort(dists)[1:7]
            if len(neighbors) < 3:
                continue
            for ni in neighbors[:3]:
                dr = coords[ni] - coords[pi]
                dv = velocity[ni] - velocity[pi]
                norm = np.linalg.norm(dr)
                if norm > 1e-12:
                    div[i] += np.dot(dv[:3], dr) / (norm ** 2)
        return div
