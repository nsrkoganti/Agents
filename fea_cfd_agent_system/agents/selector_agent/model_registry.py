"""Model registry loader — reads model_registry.yaml and returns candidate list."""

import yaml
from pathlib import Path
from loguru import logger
from agents.orchestrator.agent_state import ModelCandidate


def load_registry(registry_path: str = "configs/model_registry.yaml") -> list:
    """Load model candidates from YAML registry file."""
    path = Path(registry_path)
    if not path.exists():
        path = Path(__file__).parent.parent.parent / "configs" / "model_registry.yaml"

    with open(path) as f:
        data = yaml.safe_load(f)

    candidates = []
    for m in data.get("models", []):
        candidates.append(ModelCandidate(
            name=m["name"],
            family=m["family"],
            github_url=m["github_url"],
            install_cmd=m["install_cmd"],
            paper=m.get("paper", ""),
            mesh_requirement=m.get("mesh_requirement", "any"),
            min_data_samples=m.get("min_data_samples", 0),
            supports_field_output=m.get("supports_field_output", True),
            has_builtin_physics_loss=m.get("has_builtin_physics_loss", False),
            benchmark_l2_error=m.get("benchmark_l2_error", 0.05),
            inference_speed_score=m.get("inference_speed_score", 5),
            code_maturity_stars=m.get("code_maturity_stars", 0),
        ))

    logger.info(f"Loaded {len(candidates)} models from {path}")
    return candidates
