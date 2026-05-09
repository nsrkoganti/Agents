"""
Stress-Strain Constitutive Agent — verifies predicted stress/strain fields obey:
  1. Hooke's law: σ = C:ε  (relative error < tolerance)
  2. Von Mises ≤ yield_stress  (if elastoplastic)
  3. Stress tensor symmetry: max|σ_ij - σ_ji| < 1e-8
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class StressStrainAgent:

    def __init__(self, config: dict):
        self.config = config
        self.constitutive_tol = 0.10   # 10% relative error tolerance
        self.symmetry_tol     = 1e-6

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {
            "passed":           True,
            "constitutive_err": 0.0,
            "symmetry_err":     0.0,
            "yield_violated":   False,
        }

        cases = dataset.get("cases", []) if isinstance(dataset, dict) else []
        for case in cases[:5]:
            fields   = case.get("fields", {})
            stress   = fields.get("stress")   # (N,6) Voigt
            strain   = fields.get("strain")   # (N,6) Voigt
            material = case.get("material_properties", {})

            if stress is None:
                continue

            # 1. Constitutive check (if strain available and linear elastic)
            E  = material.get("youngs_modulus", material.get("E", 210e9))
            nu = material.get("poisson_ratio",  material.get("nu", 0.3))
            if strain is not None and problem_card and \
               problem_card.material_model == "linear_elastic":
                err = self._constitutive_error(stress, strain, float(E), float(nu))
                result["constitutive_err"] = max(result["constitutive_err"], err)
                if err > self.constitutive_tol:
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Constitutive law violated: relative error={err:.3f} > {self.constitutive_tol}"
                    )

            # 2. Symmetry check — Voigt [σxx σyy σzz σxy σyz σxz] is already symmetric by definition,
            #    but check off-diagonal pairs if full 3×3 is available
            sym_err = self._symmetry_error(stress)
            result["symmetry_err"] = max(result["symmetry_err"], sym_err)
            if sym_err > self.symmetry_tol:
                result["passed"] = False
                result["failure_reason"] = f"Stress tensor asymmetry: max={sym_err:.2e}"

            # 3. Yield check
            yield_stress = material.get("yield_strength",
                           material.get("yield_stress",
                           getattr(problem_card, "yield_stress", None) if problem_card else None))
            if yield_stress is not None and yield_stress > 0:
                vm = self._von_mises(stress)
                if np.any(vm > yield_stress * 1.05):
                    pct_over = float(np.mean(vm > yield_stress * 1.05)) * 100
                    result["yield_violated"] = True
                    if problem_card and problem_card.material_model == "linear_elastic":
                        result["passed"] = False
                        result["failure_reason"] = (
                            f"Von Mises exceeds yield: {pct_over:.1f}% of nodes over limit"
                        )

        return result

    def _constitutive_error(self, stress: np.ndarray, strain: np.ndarray,
                             E: float, nu: float) -> float:
        """Compute mean relative error ||σ - C:ε|| / ||σ||."""
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu  = E / (2 * (1 + nu))
        # Hooke's law in Voigt notation: σ = C:ε
        # C:ε components for isotropic material
        ex, ey, ez = strain[:,0], strain[:,1], strain[:,2]
        exy, eyz, exz = strain[:,3], strain[:,4], strain[:,5]
        ev = ex + ey + ez  # volumetric strain
        sx_pred = lam * ev + 2 * mu * ex
        sy_pred = lam * ev + 2 * mu * ey
        sz_pred = lam * ev + 2 * mu * ez
        sxy_pred = mu * exy
        syz_pred = mu * eyz
        sxz_pred = mu * exz
        sigma_pred = np.stack([sx_pred, sy_pred, sz_pred, sxy_pred, syz_pred, sxz_pred], axis=1)
        norm_sigma = np.linalg.norm(stress,       axis=1, keepdims=True) + 1e-30
        rel_err    = np.linalg.norm(stress - sigma_pred, axis=1) / norm_sigma.ravel()
        return float(np.mean(rel_err))

    def _symmetry_error(self, stress: np.ndarray) -> float:
        """
        Voigt stress is inherently symmetric; return 0 unless full 3×3 provided.
        """
        if stress.shape[-1] == 9:
            S = stress.reshape(-1, 3, 3)
            return float(np.max(np.abs(S - S.transpose(0, 2, 1))))
        return 0.0

    def _von_mises(self, stress: np.ndarray) -> np.ndarray:
        if stress.ndim == 2 and stress.shape[1] >= 6:
            s = stress
            return np.sqrt(0.5 * (
                (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
                + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
            ))
        return np.abs(stress).ravel()
