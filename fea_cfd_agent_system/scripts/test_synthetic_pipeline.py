"""
End-to-end pipeline test on synthetic FEA data.

Runs the full agent chain (without LLM calls where possible):
  generate data → DataAgent → AnalystAgent → [model training] → EvaluatorAgent

Usage:
    python scripts/test_synthetic_pipeline.py
    python scripts/test_synthetic_pipeline.py --cases 100 --epochs 50 --model mlp
"""

import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_fea_data import generate_dataset
from agents.data_agent.data_agent import DataAgent
from agents.analyst_agent.analyst_agent import AnalystAgent
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_torch_dataset(unified_schema: dict, target_field: str = "von_mises"):
    """
    Convert unified schema cases into PyTorch tensors for training.

    Input features per node: [x_norm, y_norm, z_norm, sigma0_norm]
      - Coordinates normalised to [-1, 1] per case
      - Applied stress (σ₀) broadcast to every node, normalised to [0, 1]

    Target: stress concentration factor SCF = field / sigma0
      This eliminates load-magnitude variation so the MLP learns geometry → SCF.
    """
    cases = unified_schema.get("cases", [])
    if not cases:
        raise ValueError("No cases in unified schema")

    X_list, Y_list, sigma0_list = [], [], []
    for case in cases:
        nodes  = case.get("nodes", np.zeros((0, 3)))    # (N, 3)
        fields = case.get("fields", {})

        if target_field not in fields:
            continue

        target = fields[target_field]                    # (N,) or (N, D)
        if target.ndim == 1:
            target = target[:, None]                     # (N, 1)

        # Applied stress per case — stored in material_properties by NumpyLoader
        mat    = case.get("material_properties", {})
        sigma0 = float(mat.get("applied_stress",
                       mat.get("applied_load",
                       case.get("applied_stress", 100e6))))
        if sigma0 == 0:
            sigma0 = 1.0

        # Normalise coordinates per case
        x_min = nodes.min(axis=0, keepdims=True)
        x_max = nodes.max(axis=0, keepdims=True)
        x_norm = 2 * (nodes - x_min) / (x_max - x_min + 1e-8) - 1   # (N, 3)

        X_list.append(x_norm.astype(np.float32))
        Y_list.append(target.astype(np.float32))
        sigma0_list.append(sigma0)

    if not X_list:
        raise ValueError(f"No cases have field '{target_field}'")

    X_coords = np.stack(X_list, axis=0)   # (C, N, 3)
    Y_raw    = np.stack(Y_list, axis=0)   # (C, N, D)
    sigma0   = np.array(sigma0_list, dtype=np.float32)   # (C,)

    # Add σ₀ as a per-node feature (broadcast along node axis)
    sigma0_norm = sigma0 / sigma0.max()   # (C,) in [0,1]
    sigma0_feat = np.broadcast_to(
        sigma0_norm[:, None, None],
        (X_coords.shape[0], X_coords.shape[1], 1)
    ).copy()
    X = np.concatenate([X_coords, sigma0_feat], axis=-1)  # (C, N, 4)

    # Target: SCF = field / sigma0 (geometry-only quantity)
    # Scale each case by its own sigma0 to remove load-magnitude variation
    Y_scf = Y_raw / sigma0[:, None, None]   # (C, N, D)
    Y_max = float(Y_scf.max())
    Y_norm = Y_scf / (Y_max + 1e-8)

    return (torch.from_numpy(X), torch.from_numpy(Y_norm),
            float(Y_max), int(Y_raw.shape[-1]))


class SimpleMLP(torch.nn.Module):
    """Baseline MLP surrogate: (N, in_dim) → (N, out_dim), applied per node."""
    def __init__(self, in_dim=4, hidden=128, n_layers=4, out_dim=1):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden), torch.nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
        layers.append(torch.nn.Linear(hidden, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        B, N, D = x.shape
        return self.net(x.view(B * N, D)).view(B, N, -1)


def train_model(X_train, Y_train, X_val, Y_val, out_dim, epochs=200,
                lr=1e-3, batch_size=16, hidden=128, n_layers=4):
    """Train an MLP surrogate and return the model + training history."""
    in_dim   = X_train.shape[-1]
    model    = SimpleMLP(in_dim=in_dim, hidden=hidden, n_layers=n_layers, out_dim=out_dim)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn  = torch.nn.MSELoss()

    n_train  = X_train.shape[0]
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_left = 30

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx  = perm[i:i + batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(idx)
        sched.step()

        epoch_loss /= n_train
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_left = 30
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:4d} | train={epoch_loss:.6f} | val={val_loss:.6f}")

    return model, history


def evaluate_model(model, X_test, Y_test, y_scale):
    """Compute R², Relative L2, and max point error (in original units)."""
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()   # (C, N, D)
    true = Y_test.numpy()              # (C, N, D)

    # Denormalise
    pred_real = pred * y_scale
    true_real = true * y_scale

    # R²
    ss_res = ((true_real - pred_real)**2).sum()
    ss_tot = ((true_real - true_real.mean())**2).sum()
    r2     = 1 - ss_res / (ss_tot + 1e-12)

    # Relative L2
    rel_l2 = np.sqrt(((true_real - pred_real)**2).sum() /
                     ((true_real**2).sum() + 1e-12))

    # Max node-wise error (fraction of true range)
    node_err  = np.abs(true_real - pred_real).max(axis=-1)   # (C, N)
    true_range = true_real.max() - true_real.min() + 1e-12
    max_pt_err = node_err.max() / true_range

    return float(r2), float(rel_l2), float(max_pt_err)


# ── Main test ─────────────────────────────────────────────────────────────────

def run_test(n_cases=200, epochs=200, model_type="mlp", target="von_mises",
             data_path=None, nx=25, ny=25):
    logger.info("=" * 60)
    logger.info("FEA SURROGATE — SYNTHETIC PIPELINE TEST")
    logger.info("=" * 60)

    # ── 1. Generate data ────────────────────────────────────────────────────
    if data_path is None:
        data_path = "data/synthetic_fea/plate_with_hole.npz"

    if not Path(data_path).exists():
        logger.info("Generating synthetic FEA data …")
        data_path = generate_dataset(n_cases=n_cases, n_x=nx, n_y=ny,
                                     output_path=data_path)
    else:
        logger.info(f"Using existing data: {data_path}")

    # ── 2. Data Agent ───────────────────────────────────────────────────────
    logger.info("\n--- DATA AGENT ---")
    config = {
        "save_dir": "/tmp/fea_test_save",
        "rag": {"enabled": False},
    }
    state = AgentSystemState(
        data_path=data_path,
        software_source="synthetic",
        run_id="synthetic_test_001",
    )

    t0 = time.time()
    data_agent = DataAgent(config)
    state = data_agent.run(state)
    logger.info(f"Data Agent: {state.data_agent_status.value} ({time.time()-t0:.2f}s)")

    if state.data_agent_status != AgentStatus.PASSED:
        logger.error(f"Data Agent FAILED: {state.error_message}")
        return False

    schema = state.unified_schema
    logger.info(f"  Cases:    {schema['n_cases']}")
    logger.info(f"  Nodes:    {schema['n_nodes']}")
    logger.info(f"  Fields:   {schema['fields']}")
    logger.info(f"  Physics:  {schema.get('physics_type', '?')}")
    logger.info(f"  Mesh:     {state.dataset.get('mesh_type', '?')}")

    # ── 3. Analyst Agent ────────────────────────────────────────────────────
    logger.info("\n--- ANALYST AGENT ---")
    analyst = AnalystAgent(config)
    state   = analyst.run(state)
    logger.info(f"Analyst Agent: {state.analyst_status.value}")

    if state.problem_card:
        pc = state.problem_card
        logger.info(f"  Physics type:    {pc.physics_type.value}")
        logger.info(f"  Mesh type:       {pc.mesh_type.value}")
        logger.info(f"  Material model:  {pc.material_model}")
        logger.info(f"  Loading type:    {pc.loading_type}")
        logger.info(f"  Element type:    {pc.element_type}")
        logger.info(f"  Data size:       {pc.data_size}")
        logger.info(f"  Nodes:           {pc.n_nodes}")
        logger.info(f"  Output targets:  {pc.output_targets}")
        logger.info(f"  R² threshold:    {pc.accuracy_threshold.get('r2', '?')}")

    # ── 4. Build PyTorch dataset ────────────────────────────────────────────
    logger.info(f"\n--- BUILDING TORCH DATASET (target={target}) ---")
    X, Y, y_scale, out_dim = build_torch_dataset(state.dataset, target_field=target)
    logger.info(f"  X shape: {tuple(X.shape)}  (C × N_nodes × 3 coordinates)")
    logger.info(f"  Y shape: {tuple(Y.shape)}  (C × N_nodes × {out_dim} outputs)")
    logger.info(f"  Y scale: {y_scale/1e6:.1f} MPa")

    # Train / val / test split (70 / 15 / 15)
    C       = X.shape[0]
    n_train = int(0.70 * C)
    n_val   = int(0.15 * C)
    perm    = torch.randperm(C)
    tr_idx  = perm[:n_train]
    va_idx  = perm[n_train:n_train + n_val]
    te_idx  = perm[n_train + n_val:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[va_idx], Y[va_idx]
    X_te, Y_te = X[te_idx], Y[te_idx]
    logger.info(f"  Train/Val/Test: {len(tr_idx)}/{len(va_idx)}/{len(te_idx)} cases")

    # ── 5. Train model ──────────────────────────────────────────────────────
    logger.info(f"\n--- TRAINING ({model_type.upper()}, {epochs} epochs) ---")
    in_dim = X.shape[-1]   # 4 = (x, y, z, sigma0)
    logger.info(f"  Input dim:  {in_dim}  (xyz + applied_stress feature)")
    t0    = time.time()
    model, history = train_model(X_tr, Y_tr, X_va, Y_va,
                                  out_dim=out_dim, epochs=epochs,
                                  hidden=128, n_layers=4)
    train_time = time.time() - t0
    logger.info(f"Training complete in {train_time:.1f}s")

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    logger.info("\n--- EVALUATION ---")
    r2, rel_l2, max_pt_err = evaluate_model(model, X_te, Y_te, y_scale)

    # Physics thresholds from analyst agent
    pc = state.problem_card
    r2_thresh      = pc.accuracy_threshold.get("r2", 0.95) if pc else 0.95
    rel_l2_thresh  = pc.accuracy_threshold.get("rel_l2", 0.03) if pc else 0.03
    max_pt_thresh  = pc.accuracy_threshold.get("max_point_error", 0.10) if pc else 0.10

    passed = (r2 >= r2_thresh and rel_l2 <= rel_l2_thresh
              and max_pt_err <= max_pt_thresh)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE RESULT")
    logger.info("=" * 60)
    logger.info(f"  R²:              {r2:.4f}  (threshold ≥ {r2_thresh:.2f})  "
                f"{'✓' if r2 >= r2_thresh else '✗'}")
    logger.info(f"  Rel L2:          {rel_l2:.4f}  (threshold ≤ {rel_l2_thresh:.2f})  "
                f"{'✓' if rel_l2 <= rel_l2_thresh else '✗'}")
    logger.info(f"  Max point error: {max_pt_err:.4f}  (threshold ≤ {max_pt_thresh:.2f})  "
                f"{'✓' if max_pt_err <= max_pt_thresh else '✗'}")
    logger.info(f"  Training time:   {train_time:.1f}s")
    logger.info(f"  Status:          {'PASSED ✓' if passed else 'FAILED ✗'}")
    logger.info("=" * 60)

    # ── 7. Quick physics sanity check ───────────────────────────────────────
    logger.info("\n--- PHYSICS SANITY CHECK ---")
    model.eval()
    case0  = state.dataset["cases"][0]
    nodes0 = case0["nodes"].astype(np.float32)        # (N, 3)
    xmin   = nodes0.min(axis=0, keepdims=True)
    xmax   = nodes0.max(axis=0, keepdims=True)
    x_norm = 2 * (nodes0 - xmin) / (xmax - xmin + 1e-8) - 1   # (N, 3)

    # Get applied stress for this case
    sigma0_c0 = float(case0.get("material_properties", {}).get("applied_stress", 100e6))
    # Build same 4D input as training: [x,y,z, sigma0_norm]
    all_sigma0 = [float(c.get("material_properties", {}).get("applied_stress", 1.0))
                  for c in state.dataset["cases"]]
    sigma0_global_max = max(all_sigma0) or 1.0
    sigma0_feat = np.full((nodes0.shape[0], 1), sigma0_c0 / sigma0_global_max, np.float32)
    x0_input    = torch.from_numpy(np.concatenate([x_norm, sigma0_feat], axis=-1)).unsqueeze(0)

    with torch.no_grad():
        # Model predicts SCF_norm = (VM/sigma0) / y_scale
        scf_norm = model(x0_input).squeeze(0).numpy()   # (N, 1)
    pred_vm = scf_norm * y_scale * sigma0_c0             # denorm SCF × sigma0

    true0 = case0["fields"].get(target, np.zeros((case0["n_nodes"], 1)))
    if true0.ndim == 1:
        true0 = true0[:, None]

    logger.info(f"  Case 0 true  VM: min={true0.min()/1e6:.2f} MPa, "
                f"max={true0.max()/1e6:.2f} MPa")
    logger.info(f"  Case 0 pred  VM: min={pred_vm.min()/1e6:.2f} MPa, "
                f"max={pred_vm.max()/1e6:.2f} MPa")
    logger.info(f"  Applied σ₀: {sigma0_c0/1e6:.1f} MPa")
    expected_scf = 3.0
    logger.info(f"  Expected SCF×σ₀: ~{expected_scf * sigma0_c0 / 1e6:.1f} MPa (Kirsch SCF=3)")
    logger.info("  Stress concentration near hole should be ~3× far-field stress")

    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic FEA pipeline test")
    parser.add_argument("--cases",  type=int,   default=200,        help="Number of FEA cases")
    parser.add_argument("--epochs", type=int,   default=200,        help="Training epochs")
    parser.add_argument("--model",  default="mlp", choices=["mlp"], help="Model type")
    parser.add_argument("--target", default="von_mises",
                        choices=["von_mises", "displacement", "stress"], help="Prediction target")
    parser.add_argument("--data",   default=None,   help="Existing .npz path (skip generation)")
    parser.add_argument("--nx",     type=int, default=25, help="Mesh x resolution")
    parser.add_argument("--ny",     type=int, default=25, help="Mesh y resolution")
    args = parser.parse_args()

    ok = run_test(
        n_cases   = args.cases,
        epochs    = args.epochs,
        model_type= args.model,
        target    = args.target,
        data_path = args.data,
        nx        = args.nx,
        ny        = args.ny,
    )
    sys.exit(0 if ok else 1)
