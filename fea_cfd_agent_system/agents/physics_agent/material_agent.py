"""
Material Physics Agent — validates predicted fields against material model constraints.

Supports:
  - Linear elastic: σ = C:ε (basic sanity only — full check in stress_strain_agent)
  - Hyperelastic Neo-Hookean: W = μ/2(I₁−3) − μln(J) + λ/2·ln²(J)
  - Elastoplastic J2: von Mises yield + isotropic hardening σ_y(εp) = σ_y0 + H·εp
  - Damage mechanics: D ∈ [0,1], σ_eff = σ/(1−D)
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class MaterialAgent:

    def __init__(self, config: dict):
        self.config = config

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {"passed": True}

        material_model = getattr(problem_card, "material_model", "linear_elastic") \
                         if problem_card else "linear_elastic"

        cases = dataset.get("cases", []) if isinstance(dataset, dict) else []
        for case in cases[:5]:
            fields   = case.get("fields", {})
            material = case.get("material_properties", {})
            stress   = fields.get("stress")
            strain   = fields.get("strain")

            # Von Mises stress
            von_mises = fields.get("von_mises") or fields.get("vonMises")
            if von_mises is None and stress is not None:
                von_mises = self._von_mises(stress)

            yield_stress = material.get("yield_strength",
                           material.get("yield_stress",
                           getattr(problem_card, "yield_stress", None) if problem_card else None))

            if material_model == "linear_elastic":
                result.update(self._check_linear_elastic(material, von_mises, yield_stress))

            elif material_model == "hyperelastic":
                result.update(self._check_hyperelastic(material, fields))

            elif material_model == "elastoplastic":
                plastic_strain = fields.get("plastic_strain") or fields.get("PEEQ")
                result.update(self._check_elastoplastic(
                    material, von_mises, yield_stress, plastic_strain))

            # Damage variable check (applies to any model)
            damage = fields.get("damage") or fields.get("D") or fields.get("SDEG")
            if damage is not None:
                result.update(self._check_damage(damage))

            if not result["passed"]:
                break

        return result

    def _check_linear_elastic(self, material, von_mises, yield_stress) -> dict:
        r = {}
        E  = material.get("youngs_modulus", material.get("E", None))
        nu = material.get("poisson_ratio",  material.get("nu", None))
        if E is not None and (float(E) <= 0 or float(E) > 1e15):
            r["passed"] = False
            r["failure_reason"] = f"Young's modulus out of physical range: E={E}"
        if nu is not None and (float(nu) < -1 or float(nu) >= 0.5):
            r["passed"] = False
            r["failure_reason"] = f"Poisson ratio out of physical range: nu={nu}"
        if von_mises is not None and yield_stress and yield_stress > 0:
            n_over = int(np.sum(von_mises > yield_stress * 1.05))
            if n_over > 0:
                r["yield_violated_nodes"] = n_over
        return r or {"passed": True}

    def _check_hyperelastic(self, material, fields) -> dict:
        mu  = material.get("shear_modulus",  material.get("mu",  None))
        lam = material.get("lame_lambda",     material.get("lam", None))
        if mu is not None and float(mu) <= 0:
            return {"passed": False, "failure_reason": f"Shear modulus μ must be > 0, got {mu}"}
        if lam is not None and float(lam) < 0:
            return {"passed": False, "failure_reason": f"Lamé λ must be ≥ 0, got {lam}"}
        # Check J (volume ratio) > 0 if available
        J = fields.get("J") or fields.get("volume_ratio")
        if J is not None and np.any(J <= 0):
            n = int(np.sum(J <= 0))
            return {"passed": False, "failure_reason": f"J ≤ 0 at {n} nodes (material inversion)"}
        return {"passed": True}

    def _check_elastoplastic(self, material, von_mises, yield_stress, plastic_strain) -> dict:
        H = material.get("hardening_modulus", material.get("H", 0.0))
        if yield_stress is None or yield_stress <= 0:
            return {"passed": True}
        if plastic_strain is not None:
            sigma_y = float(yield_stress) + float(H) * np.maximum(plastic_strain, 0)
            if von_mises is not None:
                violations = np.sum(von_mises > sigma_y * 1.05)
                if violations > 0:
                    return {
                        "passed": False,
                        "failure_reason": (
                            f"J2 yield surface violated at {violations} nodes "
                            f"(hardening H={H:.0f})"
                        ),
                    }
        return {"passed": True}

    def _check_damage(self, damage: np.ndarray) -> dict:
        if np.any(damage < -0.01) or np.any(damage > 1.01):
            return {
                "passed": False,
                "failure_reason": (
                    f"Damage variable D out of [0,1]: "
                    f"min={damage.min():.3f}, max={damage.max():.3f}"
                ),
            }
        return {"passed": True}

    def _von_mises(self, stress: np.ndarray) -> np.ndarray:
        if stress.ndim == 2 and stress.shape[1] >= 6:
            s = stress
            return np.sqrt(0.5 * (
                (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
                + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
            ))
        return np.abs(stress).ravel()
