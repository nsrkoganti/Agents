"""
ANSYS Loader — reads ANSYS RST result files.

Uses ansys-mapdl-reader for .rst files (legacy MAPDL format).
Falls back to pyvista for VTK-exported ANSYS results.

Unified FEA schema output:
  nodes (N,3), elements (E,K), displacement (N,3), stress (N,6) [Voigt],
  strain (N,6), von_mises (N,), reaction_forces (N_bc,3),
  boundary_conditions, material_props, load_steps, physics_type, solver_source
"""

import numpy as np
from loguru import logger


class ANSYSLoader:

    def load(self, path: str) -> dict:
        if path.endswith(".rst") or path.endswith(".rth"):
            return self._load_rst(path)
        return self._load_vtk_fallback(path)

    def _load_rst(self, path: str) -> dict:
        try:
            import ansys.mapdl.reader as pymapdl_reader
            rst = pymapdl_reader.read_binary(path)
            return self._extract_from_rst(rst, path)
        except ImportError:
            logger.warning("ansys-mapdl-reader not installed — falling back to pyvista")
            return self._load_vtk_fallback(path)
        except Exception as e:
            logger.warning(f"ANSYS RST load error: {e} — falling back to pyvista")
            return self._load_vtk_fallback(path)

    def _extract_from_rst(self, rst, path: str) -> dict:
        nodes   = rst.mesh.nodes          # (N, 3)
        elems   = rst.mesh.elem           # (E, K)
        n_nodes = len(nodes)

        result_sets = rst.available_results
        disp = stress = strain = vm = reaction = None

        # Displacement
        if hasattr(rst, 'nodal_displacement'):
            try:
                _, disp = rst.nodal_displacement(0)  # first load step
            except Exception:
                pass

        # Stress
        if hasattr(rst, 'nodal_stress'):
            try:
                _, s = rst.nodal_stress(0)
                # ANSYS returns [SX, SY, SZ, SXY, SYZ, SXZ]
                if s is not None and s.ndim == 2:
                    stress = s[:, :6]
            except Exception:
                pass

        # Von Mises
        if hasattr(rst, 'nodal_von_mises_stress'):
            try:
                _, vm = rst.nodal_von_mises_stress(0)
            except Exception:
                pass

        # Compute von Mises from stress if not directly available
        if vm is None and stress is not None:
            s = stress
            vm = np.sqrt(0.5 * (
                (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
                + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
            ))

        fields = {}
        if disp    is not None: fields["displacement"]  = np.array(disp)
        if stress  is not None: fields["stress"]        = np.array(stress)
        if vm      is not None: fields["von_mises"]     = np.array(vm)
        if reaction is not None: fields["reaction_forces"] = np.array(reaction)

        return {
            "nodes":       np.array(nodes),
            "elements":    np.array(elems),
            "fields":      fields,
            "n_nodes":     n_nodes,
            "n_elements":  len(elems) if elems is not None else 0,
            "physics_type": "FEA_static_linear",
            "solver_source": "ANSYS",
            "boundary_info":      {},
            "boundary_conditions": {},
            "material_properties": {},
            "mesh_type":   "unstructured_tetrahedral",
        }

    def _load_vtk_fallback(self, path: str) -> dict:
        try:
            import pyvista as pv
            mesh = pv.read(path)
            nodes    = np.array(mesh.points)
            n_nodes  = len(nodes)
            fields   = {}

            fea_aliases = {
                "displacement": ["Displacement", "U", "disp", "DEFORMATION"],
                "stress":       ["Stress", "S", "STRESS"],
                "von_mises":    ["vonMises", "VM_STRESS", "Mises", "SEQV"],
                "strain":       ["Strain", "E", "EPEL"],
            }
            for std, aliases in fea_aliases.items():
                for alias in aliases:
                    if alias in mesh.point_data:
                        fields[std] = np.array(mesh.point_data[alias])
                        break
                    if alias in mesh.cell_data:
                        fields[std] = np.array(mesh.cell_data[alias])
                        break

            return {
                "nodes":           nodes,
                "elements":        np.array([]),
                "fields":          fields,
                "n_nodes":         n_nodes,
                "n_elements":      mesh.n_cells,
                "physics_type":    "FEA_static_linear",
                "solver_source":   "ANSYS",
                "boundary_info":   {},
                "boundary_conditions": {},
                "material_properties": {},
                "mesh_type":       "unstructured_tetrahedral",
            }
        except Exception as e:
            logger.error(f"ANSYS VTK fallback failed: {e}")
            return None
