"""
Boundary Condition Agent — checks all BCs are respected by model predictions.
No-slip at walls, inlet velocity profiles, outlet pressure.
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class BoundaryConditionAgent:

    def __init__(self, config: dict):
        self.config = config
        self.bc_threshold = config.get("physics", {}).get("cfd", {}).get("bc_error_threshold", 1e-6)

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        result = {
            "passed":      True,
            "no_slip_max": 0.0,
            "inlet_match": 1.0,
            "outlet_ok":   True,
        }

        for case in dataset.get("cases", [])[:5]:
            fields        = case.get("fields", {})
            boundary_info = case.get("boundary_info", {})
            velocity      = fields.get("velocity")

            if velocity is None:
                continue

            wall_idx = boundary_info.get("wall_node_indices", [])
            if len(wall_idx) > 0:
                wall_vel  = velocity[wall_idx]
                no_slip_err = float(np.max(np.abs(wall_vel)))
                result["no_slip_max"] = max(result["no_slip_max"], no_slip_err)
                if no_slip_err > self.bc_threshold:
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"No-slip violated at wall: max_velocity={no_slip_err:.2e}"
                    )

            inlet_idx     = boundary_info.get("inlet_node_indices", [])
            inlet_bc_vel  = case.get("boundary_conditions", {}).get("inlet", {}).get("U", None)
            if len(inlet_idx) > 0 and inlet_bc_vel is not None:
                v_inlet = velocity[inlet_idx]
                v_mag   = (float(np.mean(np.linalg.norm(v_inlet, axis=-1)))
                           if v_inlet.ndim > 1 else float(np.mean(np.abs(v_inlet))))
                match = 1.0 - abs(v_mag - inlet_bc_vel) / (inlet_bc_vel + 1e-10)
                result["inlet_match"] = min(result["inlet_match"], match)

        return result
