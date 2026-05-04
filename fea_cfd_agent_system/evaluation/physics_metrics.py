"""Physics compliance metrics computed from model predictions."""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class PhysicsMetrics:
    """
    Wraps physics law checks and returns structured compliance report.
    Used by the physics agent to aggregate all checks.
    """

    def compute_cfd_compliance(self, predictions: Dict[str, np.ndarray],
                                coords: np.ndarray,
                                re_number: Optional[float] = None,
                                turbulence_model: Optional[str] = None
                                ) -> Dict[str, bool]:
        from physics.cfd_laws import CFDLaws
        from physics.turbulence_models import TurbulenceModels

        laws   = CFDLaws()
        turb   = TurbulenceModels()
        result = {}

        velocity = predictions.get("velocity")
        pressure = predictions.get("pressure")
        k        = predictions.get("tke")
        omega    = predictions.get("omega")

        if velocity is not None:
            cont_ok, div = laws.check_continuity(velocity, coords)
            result["continuity"] = cont_ok
            result["continuity_div"] = div

            vel_ok, vel_msg = laws.check_velocity_bounds(velocity, re_number)
            result["velocity_bounds"] = vel_ok

        if pressure is not None:
            p_ok, p_msg = laws.check_pressure_field(pressure)
            result["pressure_valid"] = p_ok

        if k is not None:
            k_ok, k_frac = turb.check_tke_positive(k)
            result["tke_positive"] = k_ok

        if omega is not None and k is not None:
            o_ok, o_frac = turb.check_omega_positive(omega)
            result["omega_positive"] = o_ok

            if turbulence_model in ("k-omega SST", "k-omega"):
                sst_ok, sst_issues = turb.check_k_omega_sst_bounds(k, omega)
                result["k_omega_sst"] = sst_ok
                result["k_omega_sst_issues"] = sst_issues

            if velocity is not None:
                ti_ok, ti = turb.check_turbulence_intensity(k, velocity)
                result["turbulence_intensity"] = ti_ok
                result["ti_value"] = ti

        return result

    def compute_fea_compliance(self, predictions: Dict[str, np.ndarray],
                                coords: np.ndarray) -> Dict[str, bool]:
        from physics.fea_laws import FEALaws
        from physics.material_models import MaterialModels

        laws  = FEALaws()
        mats  = MaterialModels()
        result = {}

        stress       = predictions.get("stress")
        displacement = predictions.get("displacement")

        if stress is not None:
            sym_ok, asym = laws.check_stress_tensor_symmetry(stress)
            result["stress_symmetry"] = sym_ok
            result["asymmetry_value"] = asym

            vm_ok, max_vm, frac = laws.check_von_mises_yield(stress)
            result["yield_criterion"] = vm_ok
            result["max_von_mises"]   = max_vm
            result["yield_fraction"]  = frac

        if displacement is not None:
            disp_ok, msg = laws.check_displacement_reasonableness(displacement)
            result["displacement_reasonable"] = disp_ok

        if "damage" in predictions:
            d_ok, max_d = mats.check_damage_variable(predictions["damage"])
            result["damage_valid"] = d_ok

        return result

    def aggregate_to_score(self, compliance: Dict[str, bool]) -> float:
        """Convert compliance dict to a scalar score [0, 1]."""
        bool_checks = [v for v in compliance.values() if isinstance(v, bool)]
        if not bool_checks:
            return 1.0
        return float(sum(bool_checks)) / len(bool_checks)
