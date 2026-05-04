"""Failure diagnosis utilities — explains why a model failed evaluation."""

from agents.orchestrator.agent_state import EvaluationResult


def diagnose(result: EvaluationResult, r2_min: float = 0.92,
             rel_l2_max: float = 0.05, max_pt_max: float = 0.15,
             infer_ms_max: float = 100.0) -> str:
    """Return a human-readable diagnosis string for a failed evaluation."""
    if result.r2_score < 0.5:
        return "Complete failure — model produces random predictions"
    if result.r2_score < 0.70:
        return "Severe underfitting — wrong architecture or missing physics"
    if result.r2_score < r2_min:
        return f"Moderate underfitting — R2={result.r2_score:.3f} (need {r2_min})"
    if result.rel_l2_error > rel_l2_max * 3:
        return "Extreme L2 error — model captures trends but not magnitudes"
    if result.rel_l2_error > rel_l2_max:
        return f"L2 error {result.rel_l2_error:.4f} exceeds threshold {rel_l2_max}"
    if result.max_point_error > max_pt_max:
        return f"Large local errors — max pointwise error {result.max_point_error:.3f}"
    if result.inference_time_ms > infer_ms_max:
        return f"Too slow for deployment: {result.inference_time_ms:.1f}ms"
    return "Metrics marginally below threshold"


def recommend_fix(result: EvaluationResult, r2_min: float = 0.92) -> str:
    """Return recommended fix action string."""
    if result.r2_score < 0.70:
        return "next_model"
    if result.r2_score < r2_min:
        return "tune_hyperparameters"
    if result.rel_l2_error > 0.10:
        return "increase_physics_loss"
    return "tune_hyperparameters"
