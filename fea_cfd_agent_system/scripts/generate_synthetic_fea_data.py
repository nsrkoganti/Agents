"""
Synthetic FEA Dataset Generator — Plate with Circular Hole Under Uniaxial Tension.

Physics: FEA_static_linear, unstructured tetrahedral mesh.
Analytical solution: Kirsch (1898) plane-stress stress concentration.

For applied far-field stress σ₀ in x-direction on a plate with hole of radius a:
  σ_rr = σ₀/2 (1 - a²/r²) + σ₀/2 (1 - 4a²/r² + 3a⁴/r⁴) cos2θ
  σ_θθ = σ₀/2 (1 + a²/r²) - σ₀/2 (1 + 3a⁴/r⁴) cos2θ
  σ_rθ = -σ₀/2 (1 + 2a²/r² - 3a⁴/r⁴) sin2θ
  SCF at θ=90°: σ_θθ = 3σ₀ (stress concentration factor = 3)

Displacements (plane stress, E, ν):
  u_r = σ₀r/2E [(1-ν) + (1+ν)(a²/r²)] + σ₀r/2E [(1+ν)(−1)(a²/r²) + (4−2a²/r²)] cos2θ  (approx)
  Using simpler linear approximation for small a/L ratio.

Output: data/synthetic_fea/plate_with_hole.npz
  - nodes:        (N_cases, N_nodes, 3)   — xyz per node per case
  - displacement: (N_cases, N_nodes, 3)   — ux, uy, uz
  - stress:       (N_cases, N_nodes, 6)   — Voigt σxx σyy σzz σxy σyz σxz
  - von_mises:    (N_cases, N_nodes)      — equivalent stress
  - strain:       (N_cases, N_nodes, 6)   — Voigt εxx εyy εzz εxy εyz εxz
  - elements:     (N_elements, 4)         — shared mesh connectivity
  - fixed_nodes:  (N_fixed,)              — indices of Dirichlet BC nodes
  - load_nodes:   (N_load,)              — indices of Neumann BC nodes
  - E, nu:        scalar material props
  - applied_stress: (N_cases,)           — σ₀ per case (varying load)
"""

import numpy as np
import argparse
from pathlib import Path
from loguru import logger


def plate_mesh(n_x: int = 30, n_y: int = 30, hole_radius: float = 0.1,
               plate_L: float = 1.0, plate_W: float = 1.0,
               thickness: float = 0.01, seed: int = 42) -> tuple:
    """
    Generate a 2.5D unstructured-like mesh for a plate with a central hole.
    Returns (nodes (N,3), elements (E,4), fixed_mask, load_mask).
    """
    rng = np.random.default_rng(seed)

    # Regular grid with jitter to simulate unstructured mesh
    xi = np.linspace(-plate_L/2, plate_L/2, n_x)
    yi = np.linspace(-plate_W/2, plate_W/2, n_y)
    xx, yy = np.meshgrid(xi, yi)
    xx = xx.ravel()
    yy = yy.ravel()

    # Remove nodes inside hole (plus small buffer)
    r2 = xx**2 + yy**2
    keep = r2 > (hole_radius * 1.05)**2
    xx, yy = xx[keep], yy[keep]

    # Add jitter (10% of grid spacing)
    dx = plate_L / (n_x - 1) * 0.10
    xx += rng.uniform(-dx, dx, xx.shape)
    yy += rng.uniform(-dx, dx, yy.shape)

    # Thin plate in z (one layer — 2.5D)
    zz = np.zeros_like(xx)
    nodes = np.column_stack([xx, yy, zz]).astype(np.float32)
    N = len(nodes)

    # Simple triangulation using Delaunay → split each triangle into 2 tets
    from scipy.spatial import Delaunay
    tri = Delaunay(np.column_stack([xx, yy]))
    tris = tri.simplices           # (E, 3)

    # Create a thin layer by duplicating nodes offset in z
    z_offset = thickness
    nodes_top = nodes.copy()
    nodes_top[:, 2] = z_offset
    all_nodes = np.vstack([nodes, nodes_top])   # (2N, 3)

    # Each triangle becomes 2 tets connecting bottom and top layers
    tets = []
    for t in tris:
        a, b, c = t
        at, bt, ct = a + N, b + N, c + N
        tets.append([a, b, c, at])
        tets.append([b, c, ct, at])
    elements = np.array(tets, dtype=np.int32)   # (2E, 4)

    # Boundary masks on 2N nodes
    all_N   = 2 * N
    x_all   = all_nodes[:, 0]
    y_all   = all_nodes[:, 1]

    # Fixed BC: left face (x = -plate_L/2), all DOF locked
    fixed_mask = x_all < (-plate_L/2 + plate_L * 0.03)

    # Load BC: right face (x = +plate_L/2), applied tension
    load_mask  = x_all > (plate_L/2 - plate_L * 0.03)

    return all_nodes, elements, fixed_mask, load_mask


def kirsch_fields(nodes: np.ndarray, sigma0: float, hole_radius: float,
                  E: float, nu: float) -> tuple:
    """
    Compute Kirsch analytical stress + approximate linear displacement.
    Returns stress (N,6) Voigt, displacement (N,3), strain (N,6).
    """
    x, y = nodes[:, 0], nodes[:, 1]
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    a     = hole_radius
    N     = len(nodes)

    # Avoid division by zero at hole edge; nodes should not be inside hole
    r_safe = np.maximum(r, a * 1.01)

    a2r2  = (a / r_safe)**2
    a4r4  = (a / r_safe)**4
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)

    # Polar stress components (Kirsch)
    s_rr  = sigma0/2 * (1 - a2r2) + sigma0/2 * (1 - 4*a2r2 + 3*a4r4) * cos2t
    s_tt  = sigma0/2 * (1 + a2r2) - sigma0/2 * (1 + 3*a4r4) * cos2t
    s_rt  = -sigma0/2 * (1 + 2*a2r2 - 3*a4r4) * sin2t

    # Convert polar → Cartesian
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    s_xx  = s_rr*cos_t**2 - 2*s_rt*sin_t*cos_t + s_tt*sin_t**2
    s_yy  = s_rr*sin_t**2 + 2*s_rt*sin_t*cos_t + s_tt*cos_t**2
    s_xy  = (s_rr - s_tt)*sin_t*cos_t + s_rt*(cos_t**2 - sin_t**2)
    s_zz  = np.zeros(N, np.float32)  # plane stress
    s_yz  = np.zeros(N, np.float32)
    s_xz  = np.zeros(N, np.float32)

    stress = np.column_stack([s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]).astype(np.float32)

    # Strain (Hooke's law, plane stress)
    e_xx = (s_xx - nu*s_yy) / E
    e_yy = (s_yy - nu*s_xx) / E
    e_zz = -nu*(s_xx + s_yy) / E
    e_xy = s_xy / (E / (2*(1+nu)))  # engineering shear
    e_yz = np.zeros(N, np.float32)
    e_xz = np.zeros(N, np.float32)

    strain = np.column_stack([e_xx, e_yy, e_zz, e_xy, e_yz, e_xz]).astype(np.float32)

    # Displacement: integrate strain (approximate, plane stress)
    # u_x ~ ε_xx * x + coupling; use far-field approximation
    eps0  = sigma0 / E
    u_x   = eps0 * x - nu * eps0 * a**2 * x / r_safe**2
    u_y   = -nu * eps0 * y + eps0 * a**2 * y / r_safe**2
    u_z   = np.zeros(N, np.float32)
    # Enforce u=0 on left face (approximately)
    u_x   -= u_x.min()

    displacement = np.column_stack([u_x, u_y, u_z]).astype(np.float32)

    # Von Mises
    von_mises = np.sqrt(0.5 * ((s_xx-s_yy)**2 + (s_yy-s_zz)**2 + (s_zz-s_xx)**2
                                + 6*(s_xy**2 + s_yz**2 + s_xz**2))).astype(np.float32)

    return stress, displacement, strain, von_mises


def generate_dataset(n_cases: int = 200, n_x: int = 25, n_y: int = 25,
                     noise_level: float = 0.005, seed: int = 0,
                     output_path: str = "data/synthetic_fea/plate_with_hole.npz"):
    """Generate N_cases of plate-with-hole FEA and save as batched NPZ."""
    rng = np.random.default_rng(seed)

    logger.info(f"Generating {n_cases} synthetic FEA cases ({n_x}×{n_y} mesh) …")

    # Shared mesh (same topology, nodes jittered per case slightly)
    nodes_base, elements, fixed_mask, load_mask = plate_mesh(
        n_x=n_x, n_y=n_y, hole_radius=0.1, seed=seed
    )
    N_nodes = nodes_base.shape[0]
    logger.info(f"  Mesh: {N_nodes} nodes, {elements.shape[0]} tets, "
                f"{fixed_mask.sum()} fixed BC nodes, {load_mask.sum()} load BC nodes")

    # Material parameters per case (slight variation to help generalisation)
    E_base    = 200e9       # Steel: 200 GPa
    nu_base   = 0.30
    E_vals    = rng.uniform(190e9, 210e9, n_cases)    # ±5% variation
    nu_vals   = rng.uniform(0.28,  0.32,  n_cases)
    sigma0_vals = rng.uniform(50e6, 200e6, n_cases)   # 50–200 MPa tension

    all_disp  = np.zeros((n_cases, N_nodes, 3), np.float32)
    all_stress= np.zeros((n_cases, N_nodes, 6), np.float32)
    all_strain= np.zeros((n_cases, N_nodes, 6), np.float32)
    all_vm    = np.zeros((n_cases, N_nodes),    np.float32)
    all_nodes = np.zeros((n_cases, N_nodes, 3), np.float32)

    for i in range(n_cases):
        # Slight per-case node jitter to simulate mesh variability
        jitter = rng.uniform(-0.001, 0.001, nodes_base.shape)
        nodes_i = (nodes_base + jitter).astype(np.float32)
        all_nodes[i] = nodes_i

        stress, disp, strain, vm = kirsch_fields(
            nodes_i, sigma0_vals[i], hole_radius=0.1,
            E=E_vals[i], nu=nu_vals[i]
        )

        # Add small Gaussian noise (simulate FEA discretisation error)
        noise_s = noise_level * np.abs(stress).mean() * rng.standard_normal(stress.shape)
        noise_d = noise_level * np.abs(disp).max() * rng.standard_normal(disp.shape)

        all_stress[i] = (stress + noise_s).astype(np.float32)
        all_disp[i]   = (disp + noise_d).astype(np.float32)
        all_strain[i] = strain
        all_vm[i]     = vm

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i+1}/{n_cases} cases "
                        f"(σ₀={sigma0_vals[i]/1e6:.0f} MPa, "
                        f"max VM={vm.max()/1e6:.1f} MPa)")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        nodes         = all_nodes,
        displacement  = all_disp,
        stress        = all_stress,
        strain        = all_strain,
        von_mises     = all_vm,
        elements      = elements,
        fixed_nodes   = np.where(fixed_mask)[0],
        load_nodes    = np.where(load_mask)[0],
        applied_stress= sigma0_vals.astype(np.float32),
        E             = E_vals.astype(np.float32),
        nu            = nu_vals.astype(np.float32),
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.success(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    logger.info(f"  Max Von Mises overall: {all_vm.max()/1e6:.1f} MPa "
                f"(expected ~3× max σ₀ = {3*sigma0_vals.max()/1e6:.0f} MPa)")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic FEA data generator")
    parser.add_argument("--cases",   type=int, default=200,  help="Number of cases")
    parser.add_argument("--nx",      type=int, default=25,   help="Grid x resolution")
    parser.add_argument("--ny",      type=int, default=25,   help="Grid y resolution")
    parser.add_argument("--noise",   type=float, default=0.005, help="Noise level (fraction)")
    parser.add_argument("--out",     default="data/synthetic_fea/plate_with_hole.npz")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    generate_dataset(
        n_cases=args.cases, n_x=args.nx, n_y=args.ny,
        noise_level=args.noise, seed=args.seed, output_path=args.out
    )
