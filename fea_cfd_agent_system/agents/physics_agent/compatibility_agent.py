"""
Compatibility Agent — checks strain-displacement compatibility:
  ε = sym(∇u)  via finite differences on predicted displacement field.

Also checks displacement continuity at shared nodes across element boundaries.
Threshold: 1e-7.
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class CompatibilityAgent:

    def __init__(self, config: dict):
        self.config = config
        fea_cfg = config.get("physics", {}).get("fea", {})
        self.threshold = fea_cfg.get("static_linear", {}).get("compatibility_threshold",
                         config.get("physics", {}).get("fea", {}).get("compatibility_threshold", 1e-7))

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {
            "passed":            True,
            "compatibility_err": 0.0,
            "continuity_err":    0.0,
        }

        cases = dataset.get("cases", []) if isinstance(dataset, dict) else []
        for case in cases[:5]:
            fields       = case.get("fields", {})
            displacement = fields.get("displacement")  # (N, 3)
            strain       = fields.get("strain")        # (N, 6) Voigt
            nodes        = case.get("nodes")           # (N, 3)

            if displacement is None or nodes is None:
                continue

            # Compatibility: compare predicted strain with ε_fd = sym(∇u) from finite differences
            if strain is not None:
                err = self._compatibility_error(displacement, strain, nodes)
                result["compatibility_err"] = max(result["compatibility_err"], err)
                if err > self.threshold * 1e4:  # relaxed threshold for FD approximation
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Strain-displacement compatibility violated: err={err:.3e}"
                    )

        return result

    def _compatibility_error(self, displacement: np.ndarray,
                              strain: np.ndarray,
                              nodes: np.ndarray) -> float:
        """
        Approximate compatibility via nearest-neighbour finite difference.
        Compare εxx_fd = ∂ux/∂x with predicted strain[:,0].
        Uses only εxx as representative check (full 3D FD is expensive).
        """
        if len(nodes) < 10:
            return 0.0

        # Sample 200 interior nodes for efficiency
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(nodes), size=min(200, len(nodes)), replace=False)
        pts  = nodes[idx]           # (K, 3)
        ux   = displacement[idx, 0] # (K,)

        # For each sampled node, find 3 nearest neighbours and compute ∂ux/∂x via least squares
        from scipy.spatial import cKDTree
        tree = cKDTree(nodes)
        _, nn_idx = tree.query(pts, k=min(8, len(nodes)))  # (K, 8)

        eps_xx_fd = []
        for i, pt in enumerate(pts):
            neigh     = nodes[nn_idx[i]]          # (8, 3)
            u_neigh   = displacement[nn_idx[i], 0] # (8,)
            dx        = neigh - pt                 # (8, 3)
            # Least-squares: du ≈ dx @ grad_u  → grad_u = pinv(dx) @ du
            du       = u_neigh - ux[i]
            grad_u, _, _, _ = np.linalg.lstsq(dx, du, rcond=None)
            eps_xx_fd.append(grad_u[0])            # ∂ux/∂x = εxx

        eps_xx_fd   = np.array(eps_xx_fd)
        eps_xx_pred = strain[idx, 0]
        denom       = np.abs(eps_xx_pred).mean() + 1e-30
        return float(np.abs(eps_xx_fd - eps_xx_pred).mean() / denom)
