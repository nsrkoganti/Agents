"""Fix strategy constants and helpers for the Iteration Agent."""

FIX_TYPES = [
    "tune_hyperparameters",
    "increase_physics_loss",
    "re_encode_bc",
    "next_model",
    "add_data_augmentation",
    "switch_to_pinn",
]

LAMBDA_MULTIPLIERS = {
    "continuity": 3.0,
    "bc":         5.0,
    "momentum":   2.0,
    "equilibrium": 2.0,
    "symmetry":   2.0,
}


def decide_fix_rule_based(r2: float, attempts_on_model: int,
                           bc_failed: bool, continuity_failed: bool) -> dict:
    """Rule-based fallback fix decision when LLM is unavailable."""
    if attempts_on_model >= 3:
        return {"fix_type": "next_model", "lambda_updates": {}}
    if r2 < 0.70:
        return {"fix_type": "next_model", "lambda_updates": {}}
    if bc_failed:
        return {"fix_type": "increase_physics_loss", "lambda_updates": {"bc": 10.0}}
    if continuity_failed:
        return {"fix_type": "increase_physics_loss", "lambda_updates": {"continuity": 3.0}}
    return {"fix_type": "tune_hyperparameters", "lambda_updates": {}}
