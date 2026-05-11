"""
Converts system objects into canonical text strings for embedding.
Also builds the metadata dicts stored alongside each vector.
"""

import json
from typing import Optional


def build_run_doc(run_record) -> str:
    """Text representation of a completed run for geometry_index."""
    return (
        f"physics={run_record.physics_type} "
        f"mesh={run_record.mesh_type} "
        f"data_size={run_record.data_size} "
        f"model={run_record.model_used} "
        f"success={run_record.success} "
        f"r2={run_record.r2_score:.3f} "
        f"notes={run_record.notes or ''}"
    )


def build_run_metadata(run_record) -> dict:
    return {
        "run_id":       run_record.run_id,
        "model":        run_record.model_used,
        "physics_type": run_record.physics_type,
        "mesh_type":    run_record.mesh_type,
        "data_size":    run_record.data_size,
        "r2":           run_record.r2_score,
        "rel_l2":       run_record.rel_l2,
        "success":      bool(run_record.success),
    }


def build_failure_doc(failure_record) -> str:
    """Text representation of a failure for failure_index."""
    return (
        f"model={failure_record.model_name} "
        f"physics={failure_record.physics_type} "
        f"failure={failure_record.failure_reason} "
        f"fix={failure_record.fix_tried} "
        f"r2={failure_record.r2_at_failure:.3f}"
    )


def build_failure_metadata(failure_record, r2_after: float = 0.0) -> dict:
    return {
        "model_name":     failure_record.model_name,
        "physics_type":   failure_record.physics_type,
        "failure_reason": failure_record.failure_reason,
        "fix_tried":      failure_record.fix_tried,
        "r2_before":      failure_record.r2_at_failure,
        "r2_after":       r2_after,
        "run_id":         failure_record.run_id,
    }


def build_model_perf_doc(model_name: str, physics_type: str,
                          mesh_type: str) -> str:
    """Text representation for physics_model_index."""
    return (
        f"model={model_name} "
        f"physics={physics_type} "
        f"mesh={mesh_type}"
    )


def build_model_perf_metadata(model_name: str, physics_type: str,
                               mesh_type: str, r2: float,
                               success: bool, data_size: int) -> dict:
    return {
        "model_name":   model_name,
        "physics_type": physics_type,
        "mesh_type":    mesh_type,
        "r2":           r2,
        "success":      success,
        "data_size":    data_size,
    }


def build_lambda_doc(physics_type: str, failed_checks: list,
                      lambda_json: dict) -> str:
    """Text representation for lambda_index."""
    checks  = " ".join(sorted(failed_checks)) if failed_checks else "none"
    lambdas = " ".join(f"{k}={v:.2f}" for k, v in sorted(lambda_json.items()))
    return f"physics={physics_type} failed={checks} lambdas={lambdas}"


def build_lambda_metadata(physics_type: str, failed_checks: list,
                           lambda_json: dict, r2: float, run_id: str) -> dict:
    return {
        "physics_type":  physics_type,
        "failed_checks": failed_checks,
        "lambda_json":   lambda_json,
        "r2":            r2,
        "run_id":        run_id,
    }


def build_dna_doc(dna_dict: dict, physics_type: str,
                   failed_checks: list) -> str:
    """Text representation for custom_dna_index."""
    blocks = " ".join(
        b.get("type", "") for b in dna_dict.get("core_blocks", [])
    )
    checks = " ".join(sorted(failed_checks)) if failed_checks else "none"
    return (
        f"physics={physics_type} "
        f"failed={checks} "
        f"blocks={blocks} "
        f"family={dna_dict.get('family', '')} "
        f"has_physics_loss={dna_dict.get('has_physics_loss', False)}"
    )


def build_dna_metadata(model_id: str, name: str, dna_dict: dict,
                        code: str, physics_type: str,
                        r2: float, failed_checks: list) -> dict:
    return {
        "model_id":      model_id,
        "name":          name,
        "physics_type":  physics_type,
        "core_blocks":   dna_dict.get("core_blocks", []),
        "family":        dna_dict.get("family", ""),
        "has_physics_loss": dna_dict.get("has_physics_loss", False),
        "r2":            r2,
        "failed_checks": failed_checks,
        "code":          code[:1000] if code else "",  # first 1000 chars for prompt injection
    }
