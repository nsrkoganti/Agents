"""Field extraction utilities — extracts FEA physical fields from simulation data."""

import numpy as np
from loguru import logger


class FieldExtractor:
    """Extracts and normalizes FEA physical fields from raw simulation data."""

    FEA_FIELD_ALIASES = {
        "displacement": ["Displacement", "U", "disp", "DEFORMATION", "u_total", "DISP",
                         "UX_UY_UZ", "displacement_vector"],
        "stress_xx":    ["S11", "SXX", "sigma_xx", "EPELX", "sx", "SX"],
        "stress_yy":    ["S22", "SYY", "sigma_yy", "sy", "SY"],
        "stress_zz":    ["S33", "SZZ", "sigma_zz", "sz", "SZ"],
        "stress_xy":    ["S12", "SXY", "tau_xy", "sxy", "SXY"],
        "stress_yz":    ["S23", "SYZ", "tau_yz", "syz", "SYZ"],
        "stress_xz":    ["S13", "SXZ", "tau_xz", "sxz", "SXZ"],
        "stress":       ["Stress", "S", "STRESS", "sigma"],
        "von_mises":    ["vonMises", "VM_STRESS", "Mises", "SEQV", "S_vm",
                         "von_mises_stress", "VMS", "eqv"],
        "strain_xx":    ["E11", "EXX", "EPEL_X", "ex", "EPELX"],
        "strain_yy":    ["E22", "EYY", "EPEL_Y", "ey", "EPELY"],
        "strain_zz":    ["E33", "EZZ", "EPEL_Z", "ez", "EPELZ"],
        "strain":       ["Strain", "E", "STRAIN", "epsilon", "EPEL"],
        "temperature":  ["T", "TEMP", "Temperature", "NT11", "NODAL_TEMP"],
        "reaction":     ["RF", "REACTION_FORCE", "RF1", "RF2", "RF3", "ReactionForce"],
        "pressure":     ["PHYDS", "CONTACT_PRESSURE", "CPRESS"],
        "plastic_strain": ["PEEQ", "PLASTIC_STRAIN", "EEQ", "PE"],
        "damage":       ["SDEG", "D", "DAMAGE", "damage_variable"],
    }

    def extract(self, raw_dict: dict) -> dict:
        fields     = raw_dict.get("fields", {})
        standardized = {}
        for std_name, aliases in self.FEA_FIELD_ALIASES.items():
            for alias in aliases:
                if alias in fields:
                    standardized[std_name] = fields[alias]
                    break
        return {**fields, **standardized}

    def compute_derived(self, fields: dict) -> dict:
        derived = {}

        # Build Voigt stress tensor from components if full tensor not present
        if "stress" not in fields and "stress_xx" in fields:
            components = [
                fields.get("stress_xx", np.zeros(1)),
                fields.get("stress_yy", np.zeros(1)),
                fields.get("stress_zz", np.zeros(1)),
                fields.get("stress_xy", np.zeros(1)),
                fields.get("stress_yz", np.zeros(1)),
                fields.get("stress_xz", np.zeros(1)),
            ]
            try:
                derived["stress"] = np.stack(components, axis=-1)
            except Exception:
                pass

        # Compute von Mises if not present
        if "von_mises" not in fields:
            stress = fields.get("stress") or derived.get("stress")
            if stress is not None and stress.ndim == 2 and stress.shape[1] >= 6:
                s = stress
                derived["von_mises"] = np.sqrt(0.5 * (
                    (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
                    + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
                ))

        return {**fields, **derived}
