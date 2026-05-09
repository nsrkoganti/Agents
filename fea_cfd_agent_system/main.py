"""
FEA Surrogate ML Agent System — CLI entry point.

Usage:
  python main.py --data path/to/result.rst --solver ansys --physics FEA_static_linear
  python main.py --data result.frd --solver calculix --physics FEA_dynamic
  python main.py --search-datasets --physics FEA_static_linear
  python main.py --data result.vtu --solver vtk --max-attempts 12 --output-dir results/
"""

import argparse
import sys
import os
import datetime
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(f"{log_dir}/run_{ts}.log", level="DEBUG", rotation="50 MB")


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found: {config_path} — using defaults")
        return {
            "r2_threshold":    0.92,
            "rel_l2_max":      0.05,
            "max_point_error": 0.15,
            "max_attempts":    24,
            "db_path":         "memory/experience.db",
        }
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return {}


def run_pipeline(data_path: str, config: dict,
                  output_dir: str = "results",
                  solver: str = "auto",
                  physics: str = "FEA_static_linear",
                  max_attempts: int = None,
                  search_datasets: bool = False) -> dict:
    """Run the full FEA surrogate agent pipeline."""
    from agents.orchestrator.master_orchestrator import MasterOrchestrator

    if max_attempts:
        config["max_attempts"] = max_attempts

    config["solver"]  = solver
    config["physics"] = physics

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FEA SURROGATE ML AGENT SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Data:    {data_path or '(none — searching datasets)'}")
    logger.info(f"Solver:  {solver}")
    logger.info(f"Physics: {physics}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Max iterations: {config.get('max_attempts', 24)}")
    logger.info(f"Search datasets: {search_datasets}")
    logger.info("=" * 60)

    try:
        import mlflow
        mlflow.set_experiment("fea_surrogate_agent")
        run_name = f"run_{datetime.datetime.now().strftime('%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("solver", solver)
        mlflow.log_param("physics", physics)
        mlflow.log_param("max_attempts", config.get("max_attempts", 24))
        use_mlflow = True
    except Exception:
        use_mlflow = False
        logger.warning("MLflow not available — proceeding without experiment tracking")

    try:
        orchestrator = MasterOrchestrator(config)
        final_state  = orchestrator.run(data_path, search_datasets=search_datasets)

        result = _summarize_result(final_state, output_dir)

        if use_mlflow:
            try:
                if result.get("r2"):
                    mlflow.log_metric("r2_score", result["r2"])
                if result.get("rel_l2"):
                    mlflow.log_metric("rel_l2", result["rel_l2"])
                mlflow.log_metric("n_iterations", result.get("n_iterations", 0))
                mlflow.log_param("final_model", result.get("model_name", "unknown"))
                mlflow.log_param("success", str(result.get("success", False)))
            except Exception:
                pass

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    finally:
        if use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _summarize_result(state, output_dir: str) -> dict:
    result = {
        "success":      state.pipeline_success,
        "n_iterations": state.current_attempt,
        "model_name":   None,
        "r2":           None,
        "rel_l2":       None,
        "physics_ok":   False,
        "dataset":      state.selected_dataset.get("name") if state.selected_dataset else None,
    }

    if state.selected_model:
        result["model_name"] = state.selected_model.name

    if state.evaluation_result:
        result["r2"]     = state.evaluation_result.r2_score
        result["rel_l2"] = state.evaluation_result.rel_l2_error

    if state.physics_report:
        result["physics_ok"] = (
            state.physics_report.equilibrium_passed and
            state.physics_report.boundary_conditions_passed
        )

    _print_summary(result, state)
    return result


def _print_summary(result: dict, state):
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE RESULT")
    logger.info("=" * 60)
    status = "SUCCESS" if result["success"] else "FAILED"
    logger.info(f"Status:      {status}")
    logger.info(f"Model:       {result.get('model_name', 'N/A')}")
    logger.info(f"Iterations:  {result['n_iterations']}")
    if result.get("dataset"):
        logger.info(f"Dataset:     {result['dataset']}")
    if result.get("r2") is not None:
        logger.info(f"R²:          {result['r2']:.4f}")
    if result.get("rel_l2") is not None:
        logger.info(f"Rel L2:      {result['rel_l2']:.4f}")
    logger.info(f"Physics OK:  {result['physics_ok']}")
    if state.thinking_log:
        logger.info("")
        logger.info("Agent reasoning (last 5):")
        for thought in state.thinking_log[-5:]:
            logger.info(f"  • {thought}")
    logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FEA Surrogate ML Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data model.rst --solver ansys --physics FEA_static_linear
  python main.py --data result.odb --solver abaqus --physics FEA_static_nonlinear
  python main.py --data output.frd --solver calculix --physics FEA_dynamic
  python main.py --data export.vtu --solver vtk --physics thermal
  python main.py --search-datasets --physics FEA_dynamic --output-dir results/
        """,
    )
    parser.add_argument("--data", default="",
                        help="FEA result file (.rst, .odb, .frd, .vtu, .h5, .csv, .npy)")
    parser.add_argument("--solver", default="auto",
                        choices=["ansys", "abaqus", "calculix", "starccm", "vtk", "hdf5", "auto"],
                        help="FEA solver source (default: auto-detect from file extension)")
    parser.add_argument("--physics", default="FEA_static_linear",
                        choices=["FEA_static_linear", "FEA_static_nonlinear", "FEA_dynamic",
                                 "thermal", "thermal_structural", "multiphysics"],
                        help="FEA physics type (default: FEA_static_linear)")
    parser.add_argument("--config", default="configs/base_config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save results")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Override max iteration attempts")
    parser.add_argument("--search-datasets", action="store_true",
                        help="Search and download FEA datasets from HuggingFace/GitHub/Zenodo")
    parser.add_argument("--log-dir", default="logs",
                        help="Log file directory")
    parser.add_argument("--db-path", default=None,
                        help="Override SQLite database path")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    config = load_config(args.config)
    if args.db_path:
        config["db_path"] = args.db_path

    search_datasets = getattr(args, "search_datasets", False)

    if not search_datasets and not args.data:
        logger.error("Provide --data <file> or --search-datasets")
        sys.exit(1)

    if not search_datasets and args.data and not Path(args.data).exists():
        logger.error(f"Data path does not exist: {args.data}")
        sys.exit(1)

    result = run_pipeline(
        data_path       = args.data,
        config          = config,
        output_dir      = args.output_dir,
        solver          = args.solver,
        physics         = args.physics,
        max_attempts    = args.max_attempts,
        search_datasets = search_datasets,
    )

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
