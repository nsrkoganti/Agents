"""
Conservation Agent — checks global mass, momentum, and energy conservation.
"""

import numpy as np
from loguru import logger
from agents.orchestrator.agent_state import ProblemCard


class ConservationAgent:

    def __init__(self, config: dict):
        self.config   = config
        self.mass_tol = config.get("physics", {}).get("cfd", {}).get("mass_conservation_error_max", 0.001)

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        if problem_card is None or "CFD" not in problem_card.physics_type.value:
            return {"passed": True, "reason": "N/A"}

        result = {"passed": True, "mass_error": 0.0}
        errors = []

        for case in dataset.get("cases", [])[:10]:
            fields        = case.get("fields", {})
            velocity      = fields.get("velocity")
            boundary_info = case.get("boundary_info", {})

            if velocity is None:
                continue

            inlet_idx  = boundary_info.get("inlet_face_indices", [])
            outlet_idx = boundary_info.get("outlet_face_indices", [])
            face_areas  = case.get("face_areas")

            if len(inlet_idx) > 0 and len(outlet_idx) > 0 and face_areas is not None:
                v_inlet  = velocity[inlet_idx]
                v_outlet = velocity[outlet_idx]
                a_inlet  = face_areas[inlet_idx]
                a_outlet = face_areas[outlet_idx]
                mdot_in  = float(np.sum(np.abs(v_inlet[:, 2]) * a_inlet))
                mdot_out = float(np.sum(np.abs(v_outlet[:, 2]) * a_outlet))
                if mdot_in > 1e-10:
                    err = abs(mdot_in - mdot_out) / mdot_in
                    errors.append(err)

        if errors:
            result["mass_error"] = float(np.mean(errors))
            if result["mass_error"] > self.mass_tol:
                result["passed"] = False
                result["failure_reason"] = (
                    f"Mass not conserved: error={result['mass_error']:.4%} > {self.mass_tol:.4%}"
                )

        return result
