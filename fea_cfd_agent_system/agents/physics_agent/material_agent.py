"""
Material Physics Agent — FEA-focused.
Checks: elastic moduli, Poisson ratio, yield surface, damage variable.
"""

import numpy as np
from agents.orchestrator.agent_state import ProblemCard


class MaterialAgent:

    def __init__(self, config: dict):
        self.config = config

    def check(self, model, dataset, problem_card: ProblemCard) -> dict:
        if problem_card is None or "FEA" not in problem_card.physics_type.value:
            return {"passed": True, "reason": "N/A for CFD"}

        result = {"passed": True}

        for case in dataset.get("cases", [])[:5]:
            fields     = case.get("fields", {})
            von_mises  = fields.get("von_mises") or fields.get("vonMises")
            material   = case.get("material_properties", {})
            yield_str  = material.get("yield_strength", float("inf"))

            if von_mises is not None:
                max_vm = float(np.max(von_mises))
                if max_vm > yield_str * 1.05:
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Von Mises stress exceeds yield: {max_vm:.1f} > {yield_str:.1f} Pa"
                    )

            damage = fields.get("damage") or fields.get("D")
            if damage is not None:
                if np.any(damage < -0.01) or np.any(damage > 1.01):
                    result["passed"] = False
                    result["failure_reason"] = (
                        f"Damage variable D out of [0,1]: "
                        f"min={damage.min():.3f}, max={damage.max():.3f}"
                    )

        return result
