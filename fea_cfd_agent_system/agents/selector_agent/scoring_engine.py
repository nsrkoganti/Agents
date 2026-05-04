"""
Model Scoring Engine — scores every model on 8 criteria.
Weights depend on physics type from Problem Card.
"""

from agents.orchestrator.agent_state import ModelCandidate, ProblemCard


class ModelScoringEngine:
    """Scores ModelCandidates across 8 dimensions."""

    WEIGHTS = {
        "CFD_incompressible_turbulent": {
            "physics_fit": 0.25, "mesh_fit": 0.20, "data_efficiency": 0.15,
            "accuracy": 0.15, "physics_law": 0.10, "speed": 0.08,
            "code_maturity": 0.05, "compute": 0.02,
        },
        "FEA_static_linear": {
            "physics_fit": 0.25, "mesh_fit": 0.20, "data_efficiency": 0.15,
            "accuracy": 0.15, "physics_law": 0.10, "speed": 0.05,
            "code_maturity": 0.05, "compute": 0.05,
        },
        "default": {
            "physics_fit": 0.25, "mesh_fit": 0.20, "data_efficiency": 0.15,
            "accuracy": 0.15, "physics_law": 0.10, "speed": 0.08,
            "code_maturity": 0.05, "compute": 0.02,
        },
    }

    PHYSICS_FIT = {
        "Transolver":   {"CFD_incompressible_turbulent": 9, "FEA_static_linear": 7},
        "GINO":         {"CFD_incompressible_turbulent": 8, "FEA_static_linear": 7},
        "MeshGraphNet": {"CFD_incompressible_turbulent": 8, "FEA_static_linear": 9},
        "DeepONet":     {"CFD_incompressible_turbulent": 7, "FEA_static_linear": 8},
        "FNO":          {"CFD_incompressible_turbulent": 7, "FEA_static_linear": 5},
        "PINN":         {"CFD_incompressible_turbulent": 6, "FEA_static_linear": 9},
        "PINO":         {"CFD_incompressible_turbulent": 8, "FEA_static_linear": 6},
        "GPR":          {"CFD_incompressible_turbulent": 4, "FEA_static_linear": 7},
        "XGBoost":      {"CFD_incompressible_turbulent": 3, "FEA_static_linear": 5},
    }

    def __init__(self, config: dict):
        self.config = config

    def score(self, model: ModelCandidate, problem_card: ProblemCard) -> float:
        """Score a model and store results in model.scores."""
        ptype = problem_card.physics_type.value
        weights = self.WEIGHTS.get(ptype, self.WEIGHTS["default"])

        scores = {}

        # 1. Physics fit
        fit_map = self.PHYSICS_FIT.get(model.name, {})
        scores["physics_fit"] = fit_map.get(ptype, fit_map.get("default", 5.0))

        # 2. Mesh fit
        mesh = problem_card.mesh_type.value
        req = model.mesh_requirement
        if req == "any":
            scores["mesh_fit"] = 9.0
        elif req == "unstructured" and "unstructured" in mesh:
            scores["mesh_fit"] = 10.0
        elif req == "structured" and mesh == "structured":
            scores["mesh_fit"] = 10.0
        elif req == "structured" and "unstructured" in mesh:
            scores["mesh_fit"] = 1.0
        else:
            scores["mesh_fit"] = 5.0

        # 3. Data efficiency
        n = problem_card.data_size
        min_req = model.min_data_samples
        if min_req == 0:
            scores["data_efficiency"] = 10.0
        elif n >= min_req * 3:
            scores["data_efficiency"] = 9.0
        elif n >= min_req:
            scores["data_efficiency"] = 7.0
        elif n >= min_req * 0.5:
            scores["data_efficiency"] = 4.0
        else:
            scores["data_efficiency"] = 1.0

        # 4. Accuracy (from benchmark L2 error — lower is better)
        l2 = model.benchmark_l2_error
        scores["accuracy"] = max(1.0, 10.0 - l2 * 100)

        # 5. Physics law enforcement
        scores["physics_law"] = 9.0 if model.has_builtin_physics_loss else 5.0

        # 6. Inference speed
        scores["speed"] = float(model.inference_speed_score)

        # 7. Code maturity
        stars = model.code_maturity_stars
        last = model.github_report.get("last_commit", "")
        maturity = min(8.0, stars / 500.0)
        if "2025" in last or "2026" in last:
            maturity = min(10.0, maturity + 2.0)
        scores["code_maturity"] = maturity

        # 8. Compute requirement
        if problem_card.n_nodes > 1_000_000:
            scores["compute"] = 4.0 if model.name in ["Transolver", "MS-MGN"] else 3.0
        elif problem_card.n_nodes > 100_000:
            scores["compute"] = 7.0
        else:
            scores["compute"] = 9.0

        total = sum(scores[k] * weights.get(k, 0) for k in scores)
        model.scores = scores
        model.total_score = total
        return total
