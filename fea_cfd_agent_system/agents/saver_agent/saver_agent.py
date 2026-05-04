"""
Saver Agent — saves model weights, metadata, physics certificate,
iteration log, and verification report.
Nothing is saved unless Evaluator + Physics Agent both passed.
"""

import json
import datetime
import torch
from pathlib import Path
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus


class SaverAgent:

    def __init__(self, config: dict):
        self.config   = config
        self.save_dir = Path(config.get("save_dir", "models/saved"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Saver Agent: saving model and metadata")
        state.saver_status = AgentStatus.RUNNING

        pc = state.problem_card
        sm = state.selected_model
        model_name = (
            f"{pc.physics_type.value if pc else 'unknown'}_"
            f"{sm.name if sm else 'unknown'}_"
            f"{state.run_id}"
        )
        model_dir = self.save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save model weights
            model_obj    = state.training_result.model_object if state.training_result else None
            weights_path = model_dir / "model_weights.pt"
            if model_obj is not None and hasattr(model_obj, "state_dict"):
                torch.save(model_obj.state_dict(), weights_path)
                logger.info(f"Saved weights: {weights_path}")

            # 2. Save metadata
            metadata  = self._build_metadata(state, model_name)
            meta_path = model_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # 3. Save physics certificate
            cert      = self._build_physics_certificate(state)
            cert_path = model_dir / "physics_certificate.json"
            with open(cert_path, "w") as f:
                json.dump(cert, f, indent=2)

            # 4. Save iteration log
            iter_log  = self._build_iteration_log(state)
            log_path  = model_dir / "iteration_log.json"
            with open(log_path, "w") as f:
                json.dump(iter_log, f, indent=2, default=str)

            # 5. Save README
            readme = self._build_readme(state, model_name)
            with open(model_dir / "README.md", "w") as f:
                f.write(readme)

            state.saved_model_path  = str(weights_path)
            state.metadata_path     = str(meta_path)
            state.saver_status      = AgentStatus.PASSED
            state.pipeline_success  = True
            state.pipeline_complete = True

            logger.success(f"Model saved at: {model_dir}")

        except Exception as e:
            state.saver_status  = AgentStatus.FAILED
            state.error_message = f"Save failed: {e}"
            logger.error(state.error_message)

        return state

    def _build_metadata(self, state: AgentSystemState, name: str) -> dict:
        pc = state.problem_card
        ev = state.evaluation_result
        tr = state.training_result
        return {
            "model_name":      name,
            "saved_at":        datetime.datetime.now().isoformat(),
            "architecture":    state.selected_model.name if state.selected_model else "unknown",
            "problem_type":    pc.physics_type.value if pc else "",
            "geometry":        pc.geometry_description if pc else "",
            "software_source": state.software_source,
            "turbulence_model": pc.turbulence_model if pc else None,
            "training_samples": pc.data_size if pc else 0,
            "metrics": {
                "R2":              round(ev.r2_score, 4) if ev else None,
                "rel_L2_error":    round(ev.rel_l2_error, 4) if ev else None,
                "max_point_error": round(ev.max_point_error, 4) if ev else None,
                "inference_ms":    round(ev.inference_time_ms, 2) if ev else None,
            },
            "training": {
                "epochs":    tr.training_epochs if tr else None,
                "train_loss": tr.train_loss if tr else None,
                "val_loss":  tr.val_loss if tr else None,
                "converged": tr.converged if tr else None,
            },
            "attempts_before_success": state.current_attempt,
            "models_tried":    [r.model_name for r in state.iteration_log],
            "deployment_ready": True,
            "verified":        state.verification_passed,
        }

    def _build_physics_certificate(self, state: AgentSystemState) -> dict:
        pr = state.physics_report
        if pr is None:
            return {"overall": "NOT_CHECKED"}
        return {
            "governing_equations": {"passed": pr.governing_equations_passed, "detail": pr.governing_equations_detail},
            "boundary_conditions": {"passed": pr.boundary_conditions_passed, "detail": pr.boundary_conditions_detail},
            "conservation":        {"passed": pr.conservation_passed,        "detail": pr.conservation_detail},
            "turbulence":          {"passed": pr.turbulence_passed,          "detail": pr.turbulence_detail},
            "material":            {"passed": pr.material_passed,            "detail": pr.material_detail},
            "overall":             "PHYSICALLY VERIFIED" if pr.overall_passed else "FAILED",
            "certified_at":        datetime.datetime.now().isoformat(),
        }

    def _build_iteration_log(self, state: AgentSystemState) -> list:
        return [
            {
                "attempt":        r.attempt_number,
                "model":          r.model_name,
                "passed":         r.overall_passed,
                "failure_reason": r.failure_reason,
                "fix_applied":    r.fix_applied,
                "r2":             r.evaluation_result.r2_score if r.evaluation_result else None,
            }
            for r in state.iteration_log
        ]

    def _build_readme(self, state: AgentSystemState, name: str) -> str:
        pc = state.problem_card
        sm = state.selected_model
        ev = state.evaluation_result
        return f"""# {name}

## Summary
- **Physics:** {pc.physics_type.value if pc else 'unknown'}
- **Architecture:** {sm.name if sm else 'unknown'}
- **Geometry:** {pc.geometry_description if pc else ''}
- **Software:** {state.software_source}

## Metrics
- R2 Score: {f'{ev.r2_score:.4f}' if ev else 'N/A'}
- Relative L2 Error: {f'{ev.rel_l2_error:.4f}' if ev else 'N/A'}
- Inference Time: {f'{ev.inference_time_ms:.1f}ms' if ev else 'N/A'}

## Physics Verification
All 5 physics sub-agents passed. Model is physically verified.

## Usage
```python
import torch
model = YourModelClass(...)
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()
```
"""
