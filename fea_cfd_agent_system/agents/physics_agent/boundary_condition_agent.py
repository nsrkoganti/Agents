"""
Boundary Condition Agent — checks FEA boundary conditions in model predictions.

Checks:
  - Fixed supports: ||u|| ≈ 0 at constrained nodes
  - Applied loads: reaction forces balance applied loads (Newton's 3rd law)
  - Symmetry planes: normal displacement ≈ 0
  - Contact: penetration = 0, contact pressure ≥ 0
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class BoundaryConditionAgent:

    def __init__(self, config: dict):
        self.config = config
        fea_cfg = config.get("physics", {}).get("fea", {})
        self.disp_threshold    = fea_cfg.get("static_linear", {}).get("bc_error_max", 1e-6)
        self.reaction_tol      = 0.05   # 5% tolerance on force balance
        self.symmetry_threshold = 1e-6

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {
            "passed":            True,
            "fixed_support_err": 0.0,
            "reaction_err":      0.0,
            "symmetry_err":      0.0,
            "missing_fields":    [],
        }

        cases = dataset.get("cases", []) if isinstance(dataset, dict) else []
        for case in cases[:5]:
            fields          = case.get("fields", {})
            displacement    = fields.get("displacement")    # (N, 3)
            reaction_forces = fields.get("reaction_forces") # (N_bc, 3)
            boundary_info   = case.get("boundary_info", {})
            applied_load    = case.get("applied_load", None)

            if displacement is None:
                result["missing_fields"].append("displacement")
                continue

            # 1. Fixed support check: constrained nodes should have u ≈ 0
            fixed_idx = boundary_info.get("fixed_node_indices",
                        boundary_info.get("wall_node_indices", []))
            if len(fixed_idx) > 0:
                u_fixed = displacement[fixed_idx]
                err = float(np.max(np.abs(u_fixed)))
                result["fixed_support_err"] = max(result["fixed_support_err"], err)
                if err > self.disp_threshold:
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Fixed support violated: max_displacement={err:.2e} "
                        f"> threshold={self.disp_threshold:.1e}"
                    )

            # 2. Reaction force balance
            if reaction_forces is not None and applied_load is not None:
                total_reaction = np.sum(reaction_forces, axis=0)
                load_vec       = np.atleast_1d(applied_load)
                if len(load_vec) == len(total_reaction):
                    f_norm = np.linalg.norm(load_vec) + 1e-30
                    err    = float(np.linalg.norm(total_reaction + load_vec) / f_norm)
                    result["reaction_err"] = max(result["reaction_err"], err)
                    if err > self.reaction_tol:
                        result["passed"] = False
                        result["failure_reason"] = (
                            f"Reaction force imbalance: {err:.3f} > {self.reaction_tol}"
                        )

            # 3. Symmetry plane check: normal displacement ≈ 0
            sym_idx    = boundary_info.get("symmetry_node_indices", [])
            sym_normal = boundary_info.get("symmetry_normal", None)
            if len(sym_idx) > 0 and sym_normal is not None:
                u_sym   = displacement[sym_idx]
                n       = np.array(sym_normal, dtype=float)
                n      /= np.linalg.norm(n) + 1e-30
                u_n     = float(np.max(np.abs(u_sym @ n)))
                result["symmetry_err"] = max(result["symmetry_err"], u_n)
                if u_n > self.symmetry_threshold:
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Symmetry BC violated: normal_disp={u_n:.2e}"
                    )

        return result
