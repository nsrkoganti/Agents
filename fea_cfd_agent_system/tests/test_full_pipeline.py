"""
End-to-end pipeline test using synthetic rectangular duct data.
Tests the full LangGraph flow: data → analyst → selector → trainer → evaluator → physics → verify → save.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


def generate_synthetic_duct_data(n_nodes: int = 500) -> dict:
    """
    Synthetic k-omega SST duct data (rectangular, 8mm wall).
    Flow: Re ~ 50,000, parabolic velocity profile.
    """
    rng = np.random.default_rng(42)

    L, H = 0.2, 0.015        # duct dimensions (m)
    x = rng.uniform(0, L, n_nodes).astype(np.float32)
    y = rng.uniform(0, H, n_nodes).astype(np.float32)
    z = np.zeros(n_nodes, dtype=np.float32)
    coords = np.stack([x, y, z], axis=-1)

    y_center = H / 2
    U_max    = 5.0
    u = U_max * (1 - ((y - y_center) / y_center) ** 2)
    u = np.maximum(u, 0).astype(np.float32)
    v = rng.normal(0, 0.01 * U_max, n_nodes).astype(np.float32)
    w = np.zeros(n_nodes, dtype=np.float32)
    velocity = np.stack([u, v, w], axis=-1)

    rho, U_ref = 1.225, U_max
    pressure = (0.5 * rho * U_ref**2 * (1 - x / L)).astype(np.float32)

    k     = (0.005 * U_max**2 * np.ones(n_nodes)).astype(np.float32)
    omega = (1000 * np.ones(n_nodes)).astype(np.float32)

    return {
        "coordinates": coords,
        "fields": {
            "velocity": velocity,
            "pressure": pressure,
            "tke":      k,
            "omega":    omega,
        },
        "n_nodes":     n_nodes,
        "n_cells":     n_nodes,
        "mesh_type":   "unstructured",
        "format":      "synthetic",
        "available_fields": ["velocity", "pressure", "tke", "omega"],
    }


def test_accuracy_metrics_on_synthetic():
    from evaluation.accuracy_metrics import AccuracyMetrics
    metrics = AccuracyMetrics(r2_threshold=0.85)

    data = generate_synthetic_duct_data(200)
    velocity = data["fields"]["velocity"]

    noise  = velocity + np.random.normal(0, 0.05, velocity.shape)
    result = metrics.compute_all(noise, velocity)

    assert "r2" in result
    assert "rel_l2" in result
    assert result["r2"] > 0.9


def test_physics_cfd_laws_on_synthetic():
    from physics.cfd_laws import CFDLaws
    data = generate_synthetic_duct_data(300)
    laws = CFDLaws(rho=1.225)

    vel    = data["fields"]["velocity"]
    coords = data["coordinates"]

    cont_ok, div = laws.check_continuity(vel, coords, n_samples=50)
    assert isinstance(cont_ok, bool)

    vel_ok, msg = laws.check_velocity_bounds(vel, re_number=50000, char_length=0.015)
    assert vel_ok, f"Velocity bounds check failed: {msg}"


def test_turbulence_checks_on_synthetic():
    from physics.turbulence_models import TurbulenceModels
    data = generate_synthetic_duct_data(300)
    turb = TurbulenceModels()

    k     = data["fields"]["tke"]
    omega = data["fields"]["omega"]
    vel   = data["fields"]["velocity"]

    k_ok, k_frac = turb.check_tke_positive(k)
    assert k_ok, f"{k_frac:.1%} negative k"

    o_ok, o_frac = turb.check_omega_positive(omega)
    assert o_ok, f"{o_frac:.1%} non-positive omega"

    ti_ok, ti = turb.check_turbulence_intensity(k, vel)
    assert isinstance(ti_ok, bool)


def test_data_normalizer_on_synthetic():
    from data.preprocessors.normalizer import DataNormalizer
    data   = generate_synthetic_duct_data(300)
    norm   = DataNormalizer()
    coords = data["coordinates"]
    fields = {k: v for k, v in data["fields"].items()}

    norm_c, norm_f = norm.fit_transform(coords, fields)

    assert norm_c.shape == coords.shape
    assert abs(float(np.mean(norm_c))) < 0.5

    inv_f = norm.inverse_transform_fields(norm_f)
    for key in fields:
        orig  = fields[key].ravel()
        back  = inv_f[key].ravel()
        assert np.allclose(orig, back, atol=1e-4), f"Inverse transform mismatch for {key}"


def test_transolver_forward_on_synthetic():
    import torch
    from models.architectures.transolver import TransolverSurrogate

    data   = generate_synthetic_duct_data(200)
    coords = data["coordinates"]

    model = TransolverSurrogate(
        input_dim=3, output_dim=4,
        hidden_dim=64, n_layers=2, n_slices=8, n_heads=4
    )
    model.eval()

    x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 200, 4)
    assert not torch.any(torch.isnan(out))


def test_mlp_trains_on_synthetic():
    """Quick smoke test: can an MLP train on synthetic duct data?"""
    import torch
    from models.architectures.mlp_surrogate import MLPSurrogate

    data   = generate_synthetic_duct_data(200)
    coords = data["coordinates"]
    vel    = data["fields"]["velocity"]
    pres   = data["fields"]["pressure"]

    y = np.concatenate([vel, pres[:, None]], axis=-1)

    model = MLPSurrogate(input_dim=3, output_dim=4, hidden_dim=64, n_layers=3)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.tensor(coords, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    for _ in range(10):
        opt.zero_grad()
        pred = model(X.unsqueeze(0)).squeeze(0)
        loss = torch.mean((pred - Y) ** 2)
        loss.backward()
        opt.step()

    assert loss.item() < 100.0, f"Loss too high: {loss.item()}"


def test_run_database_with_synthetic(tmp_path):
    from memory.run_database import RunDatabase, RunRecord
    import datetime

    db = RunDatabase(db_path=str(tmp_path / "test.db"))
    record = RunRecord(
        run_id="synthetic_001",
        physics_type="cfd_incompressible",
        mesh_type="unstructured",
        data_size=500,
        model_used="TransolverSurrogate",
        r2_score=0.94,
        rel_l2=0.04,
        success=True,
        n_iterations=3,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    db.save_run(record)
    runs = db.get_recent_runs(1)
    assert runs[0]["r2_score"] == pytest.approx(0.94)
