"""
Physics Master Agent — runs 5 FEA physics sub-agents in parallel.
A model CANNOT be saved unless it passes ALL physics checks.
"""

import concurrent.futures
import datetime
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus, PhysicsReport
from agents.physics_agent.equilibrium_agent      import EquilibriumAgent
from agents.physics_agent.stress_strain_agent    import StressStrainAgent
from agents.physics_agent.compatibility_agent    import CompatibilityAgent
from agents.physics_agent.boundary_condition_agent import BoundaryConditionAgent
from agents.physics_agent.material_agent         import MaterialAgent


class PhysicsMasterAgent:
    """
    Runs 5 FEA physics sub-agents simultaneously via ThreadPoolExecutor.
    Aggregates results into PhysicsReport.
    Provides fix instructions to Iteration Agent on failure.
    """

    def __init__(self, config: dict):
        self.config             = config
        self.equilibrium_agent  = EquilibriumAgent(config)
        self.stress_strain_agent = StressStrainAgent(config)
        self.compatibility_agent = CompatibilityAgent(config)
        self.bc_agent           = BoundaryConditionAgent(config)
        self.material_agent     = MaterialAgent(config)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Physics Master: running 5 FEA sub-agents in parallel")
        state.physics_status = AgentStatus.RUNNING

        if state.training_result is None or state.training_result.model_object is None:
            logger.warning("Physics Agent: no model to check — skipping")
            state.physics_report = PhysicsReport(overall_passed=True)
            state.physics_status = AgentStatus.PASSED
            return state

        model   = state.training_result.model_object
        dataset = state.dataset
        pc      = state.problem_card
        report  = PhysicsReport()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                "equilibrium":   ex.submit(self.equilibrium_agent.check,   model, dataset, pc),
                "stress_strain": ex.submit(self.stress_strain_agent.check, model, dataset, pc),
                "compatibility": ex.submit(self.compatibility_agent.check, model, dataset, pc),
                "bc":            ex.submit(self.bc_agent.check,            model, dataset, pc),
                "material":      ex.submit(self.material_agent.check,      model, dataset, pc),
            }
            results = {k: f.result() for k, f in futures.items()}

        report.equilibrium_passed   = results["equilibrium"]["passed"]
        report.equilibrium_detail   = results["equilibrium"]
        report.stress_strain_passed = results["stress_strain"]["passed"]
        report.stress_strain_detail = results["stress_strain"]
        report.compatibility_passed = results["compatibility"]["passed"]
        report.compatibility_detail = results["compatibility"]
        report.boundary_conditions_passed = results["bc"]["passed"]
        report.boundary_conditions_detail = results["bc"]
        report.material_passed      = results["material"]["passed"]
        report.material_detail      = results["material"]

        report.overall_passed = all([
            report.equilibrium_passed,
            report.stress_strain_passed,
            report.compatibility_passed,
            report.boundary_conditions_passed,
            report.material_passed,
        ])

        if not report.overall_passed:
            report.fix_instructions = self._build_fix_instructions(results)
            report.physics_lambda_updates = self._build_lambda_updates(results, state)

        state.physics_report = report
        state.physics_status = AgentStatus.PASSED if report.overall_passed else AgentStatus.FAILED

        if not report.boundary_conditions_passed:
            self._publish_insufficient_bc(state, results["bc"])

        logger.info(
            f"Physics: equilibrium={report.equilibrium_passed}, "
            f"stress_strain={report.stress_strain_passed}, "
            f"compatibility={report.compatibility_passed}, "
            f"bc={report.boundary_conditions_passed}, "
            f"material={report.material_passed}, "
            f"overall={report.overall_passed}"
        )
        return state

    def _build_fix_instructions(self, results: dict) -> str:
        fixes = []
        if not results["equilibrium"]["passed"]:
            res = results["equilibrium"].get("residual_max", "?")
            fixes.append(f"Increase lambda_equilibrium: residual={res:.3e}")
        if not results["stress_strain"]["passed"]:
            err = results["stress_strain"].get("constitutive_err", "?")
            fixes.append(f"Enforce constitutive law: constitutive_err={err:.3f}")
        if not results["compatibility"]["passed"]:
            err = results["compatibility"].get("compatibility_err", "?")
            fixes.append(f"Add compatibility loss: compat_err={err:.3e}")
        if not results["bc"]["passed"]:
            err = results["bc"].get("fixed_support_err", "?")
            fixes.append(f"Re-encode BC features: fixed_support_err={err:.2e}")
        if not results["material"]["passed"]:
            reason = results["material"].get("failure_reason", "check material model")
            fixes.append(f"Material violation: {reason}")
        return " | ".join(fixes)

    def _build_lambda_updates(self, results: dict, state: AgentSystemState) -> dict:
        lambdas = {}
        w = state.physics_lambda_weights
        if not results["equilibrium"]["passed"]:
            lambdas["equilibrium"] = w.get("equilibrium", 1.0) * 2.0
        if not results["stress_strain"]["passed"]:
            lambdas["constitutive"] = w.get("constitutive", 1.0) * 2.0
        if not results["compatibility"]["passed"]:
            lambdas["compatibility"] = w.get("compatibility", 1.0) * 2.0
        if not results["bc"]["passed"]:
            lambdas["bc"] = w.get("bc", 2.0) * 5.0
        return lambdas

    def _publish_insufficient_bc(self, state: AgentSystemState, bc_result: dict) -> None:
        already = any(
            m.get("type") == "INSUFFICIENT_BC_DATA" and not m.get("handled")
            for m in state.agent_messages
        )
        if already:
            return
        state.agent_messages.append({
            "from":      "physics_agent",
            "to":        "dataset_agent",
            "type":      "INSUFFICIENT_BC_DATA",
            "missing":   bc_result.get("missing_fields", []),
            "reason":    f"BC check failed: {bc_result.get('failure_reason', 'unknown')}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "handled":   False,
        })
        logger.info("Published INSUFFICIENT_BC_DATA to agent message bus")
