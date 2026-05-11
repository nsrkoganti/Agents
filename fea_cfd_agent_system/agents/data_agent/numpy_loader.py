"""
NumpyLoader — loads .npz / .npy FEA datasets into the unified schema.

The .npz file is expected to contain either:
  - A 'cases' array (object array of dicts) for multi-case datasets, OR
  - Flat arrays (nodes, displacement, stress, …) for single-case files, OR
  - Stacked arrays (displacement[N_cases, N_nodes, 3]) for batch files.
"""

import numpy as np
from pathlib import Path
from loguru import logger


_FIELD_ALIASES = {
    "displacement": ["displacement", "disp", "U", "DISP", "u", "deformation",
                     "displacement_vector", "UX_UY_UZ"],
    "stress":       ["stress", "STRESS", "S", "sigma", "stress_voigt"],
    "strain":       ["strain", "STRAIN", "E", "epsilon", "strain_voigt"],
    "von_mises":    ["von_mises", "vonMises", "VM_STRESS", "SEQV", "Mises",
                     "S_vm", "VMS", "eqv", "von_mises_stress"],
    "temperature":  ["temperature", "T", "TEMP", "NT11"],
    "reaction":     ["reaction", "RF", "reaction_forces", "REACTION_FORCE"],
    "pressure":     ["pressure", "PHYDS", "CONTACT_PRESSURE"],
    "plastic_strain": ["plastic_strain", "PEEQ", "plastic_strain_eq"],
}

_GEOMETRY_KEYS  = {"nodes", "coordinates", "coords", "node_coords", "xyz", "positions"}
_ELEMENT_KEYS   = {"elements", "connectivity", "element_connectivity",
                   "cells", "elem", "tets", "hexs"}
_MATERIAL_KEYS  = {"E", "nu", "yield_stress", "density", "rho", "shear_modulus",
                   "poisson", "young_modulus", "elastic_modulus",
                   "applied_stress", "applied_load", "load_magnitude"}
_BC_KEYS        = {"fixed_nodes", "load_nodes", "symmetry_nodes",
                   "dirichlet_nodes", "neumann_nodes", "bc_nodes"}


def _resolve_field(npz_keys: set, target: str, aliases: list):
    """Return the first matching key from npz_keys, or None."""
    if target in npz_keys:
        return target
    for a in aliases:
        if a in npz_keys:
            return a
    return None


def _build_case(data: dict) -> dict:
    """
    Convert a flat dict of arrays (keys → np arrays) into one unified case dict.
    """
    keys = set(data.keys())

    # ── Geometry ────────────────────────────────────────────────────────────
    nodes_key = next((k for k in _GEOMETRY_KEYS if k in keys), None)
    nodes     = np.asarray(data[nodes_key]) if nodes_key else None
    if nodes is not None and nodes.ndim == 1:
        nodes = nodes.reshape(-1, 3)          # flat (3N,) → (N,3)

    elem_key  = next((k for k in _ELEMENT_KEYS if k in keys), None)
    elements  = np.asarray(data[elem_key]) if elem_key else np.empty((0, 4), dtype=int)

    n_nodes   = nodes.shape[0] if nodes is not None else 0

    # ── Physical fields ──────────────────────────────────────────────────────
    fields = {}
    for std_name, aliases in _FIELD_ALIASES.items():
        key = _resolve_field(keys, std_name, aliases)
        if key is not None:
            arr = np.asarray(data[key], dtype=np.float32)
            fields[std_name] = arr
            if n_nodes == 0 and arr.ndim >= 1:
                n_nodes = arr.shape[0]

    # Compute von Mises from stress Voigt if missing
    if "von_mises" not in fields and "stress" in fields:
        s = fields["stress"]
        if s.ndim == 2 and s.shape[1] >= 6:
            sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
            sxy, syz, sxz = s[:, 3], s[:, 4], s[:, 5]
            vm = np.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                                 + 6*(sxy**2 + syz**2 + sxz**2)))
            fields["von_mises"] = vm.astype(np.float32)

    # ── Material properties ───────────────────────────────────────────────
    material = {}
    for mk in _MATERIAL_KEYS:
        if mk in data:
            v = data[mk]
            material[mk] = float(v) if np.asarray(v).ndim == 0 else np.asarray(v).tolist()

    # ── Boundary conditions ───────────────────────────────────────────────
    bc_info = {}
    for bk in _BC_KEYS:
        if bk in data:
            bc_info[bk] = np.asarray(data[bk]).tolist()

    # ── Detect element type from connectivity shape ───────────────────────
    eshape   = elements.shape[-1] if elements.ndim == 2 and elements.shape[0] > 0 else 4
    etype_map = {4: "tet", 8: "hex", 3: "shell", 2: "beam"}
    elem_type = etype_map.get(eshape, "tet")
    mesh_type_map = {"tet": "unstructured_tetrahedral", "hex": "unstructured_hexahedral",
                     "shell": "unstructured_polyhedral", "beam": "tabular"}
    mesh_type = mesh_type_map.get(elem_type, "unstructured_tetrahedral")

    # ── Physics type from scalar keys ────────────────────────────────────
    physics_type = "FEA_static_linear"
    if "plastic_strain" in fields or "damage" in fields:
        physics_type = "FEA_static_nonlinear"
    elif "temperature" in fields and "displacement" in fields:
        physics_type = "thermal_structural"
    elif "temperature" in fields:
        physics_type = "thermal"

    # ── Load steps ───────────────────────────────────────────────────────
    load_steps = int(data.get("load_steps", data.get("n_load_steps", 1)))
    if load_steps > 5:
        physics_type = "FEA_dynamic"

    return {
        "nodes":           nodes if nodes is not None else np.zeros((n_nodes, 3), np.float32),
        "elements":        elements,
        "fields":          fields,
        "n_nodes":         n_nodes,
        "n_elements":      elements.shape[0] if elements.ndim == 2 else 0,
        "physics_type":    physics_type,
        "solver_source":   str(data.get("solver_source", "synthetic")),
        "mesh_type":       mesh_type,
        "element_type":    elem_type,
        "boundary_info":   bc_info,
        "material_properties": material,
        "load_steps":      load_steps,
        "mesh_quality":    {"skewness_max": float(data.get("skewness_max", 0.0))},
        "boundary_types":  list(bc_info.keys()),
    }


class NumpyLoader:
    """
    Loads .npz or .npy FEA datasets.

    Supported layouts:
      1. Multi-case batch: NPZ with 'nodes' (C,N,3), 'displacement' (C,N,3), etc.
      2. Single-case:      NPZ with 'nodes' (N,3), 'displacement' (N,3), etc.
      3. Object-array:     NPZ with 'cases' key storing an array of dicts.
    """

    def load(self, path: str) -> dict | None:
        p = Path(path)
        try:
            if p.suffix == ".npy":
                arr = np.load(str(p), allow_pickle=True)
                if arr.dtype == object:
                    cases = [_build_case(c) for c in arr]
                else:
                    cases = [_build_case({"displacement": arr})]
                return self._wrap(cases, path)

            # .npz
            raw = np.load(str(p), allow_pickle=True)
            data = dict(raw)

            # Layout 1: 'cases' object array
            if "cases" in data and data["cases"].dtype == object:
                cases = [_build_case(c) for c in data["cases"]]
                return self._wrap(cases, path)

            # Layout 2: stacked batch arrays — first axis is case index
            # Heuristic: if any field has ndim==3, it's (C, N, D)
            is_batch = any(
                np.asarray(v).ndim == 3
                for k, v in data.items()
                if k not in _GEOMETRY_KEYS and k not in _ELEMENT_KEYS
            )
            if is_batch:
                n_cases = None
                for v in data.values():
                    a = np.asarray(v)
                    if a.ndim >= 2:
                        n_cases = a.shape[0]
                        break
                if n_cases:
                    cases = []
                    for i in range(n_cases):
                        case_dict = {}
                        for k, v in data.items():
                            a = np.asarray(v)
                            if a.ndim >= 2:
                                case_dict[k] = a[i]
                            elif a.ndim == 1 and len(a) == n_cases:
                                # Per-case 1-D array (e.g. applied_stress, E, nu)
                                case_dict[k] = a[i]
                            else:
                                case_dict[k] = a   # shared scalar / array
                        cases.append(_build_case(case_dict))
                    return self._wrap(cases, path)

            # Layout 3: single-case flat NPZ
            case = _build_case(data)
            return self._wrap([case], path)

        except Exception as e:
            logger.error(f"NumpyLoader failed on {path}: {e}")
            return None

    @staticmethod
    def _wrap(cases: list, path: str) -> list:
        """Return a flat list of case dicts — DataAgent collects these directly."""
        if not cases:
            return []
        s = cases[0]
        logger.info(f"NumpyLoader: {len(cases)} cases, "
                    f"{s['n_nodes']} nodes, fields={list(s['fields'].keys())}")
        return cases
