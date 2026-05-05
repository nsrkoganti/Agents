"""
Physics Master Agent — runs all 5 physics sub-agents in parallel.
A model CANNOT be saved unless it passes ALL physics checks.
"""

import concurrent.futures
import datetime
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus, PhysicsReport
from agents.physics_agent.governing_equation_agent import GoverningEquationAgent
from agents.physics_agent.boundary_condition_agent import BoundaryConditionAgent
from agents.physics_agent.conservation_agent import ConservationAgent
from agents.physics_agent.turbulence_agent import TurbulenceAgent
from agents.physics_agent.material_agent import MaterialAgent


class PhysicsMasterAgent:
    """
    Runs 5 physics sub-agents simultaneously.
    Aggregates results into PhysicsReport.
    Provides fix instructions to Iteration Agent on failure.
    """

    def __init__(self, config: dict):
        self.config       = config
        self.gov_eq_agent = GoverningEquationAgent(config)
        self.bc_agent     = BoundaryConditionAgent(config)
        self.conservation = ConservationAgent(config)
        self.turbulence   = TurbulenceAgent(config)
        self.material     = MaterialAgent(config)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Physics Master: running 5 sub-agents")
        state.physics_status = AgentStatus.RUNNING

        if state.training_result is None or state.training_result.model_object is None:
            logger.warning("Physics Agent: no model to check — skipping")
            report = PhysicsReport(overall_passed=True)
            state.physics_report = report
            state.physics_status = AgentStatus.PASSED
            return state

        model   = state.training_result.model_object
        dataset = state.dataset
        pc      = state.problem_card

        report = PhysicsReport()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                "gov_eq":       ex.submit(self.gov_eq_agent.check,  model, dataset, pc),
                "bc":           ex.submit(self.bc_agent.check,       model, dataset, pc),
                "conservation": ex.submit(self.conservation.check,   model, dataset, pc),
                "turbulence":   ex.submit(self.turbulence.check,     model, dataset, pc),
                "material":     ex.submit(self.material.check,       model, dataset, pc),
            }
            results = {k: f.result() for k, f in futures.items()}

        report.governing_equations_passed = results["gov_eq"]["passed"]
        report.governing_equations_detail = results["gov_eq"]
        report.boundary_conditions_passed = results["bc"]["passed"]
        report.boundary_conditions_detail = results["bc"]
        report.conservation_passed        = results["conservation"]["passed"]
        report.conservation_detail        = results["conservation"]
        report.turbulence_passed          = results["turbulence"]["passed"]
        report.turbulence_detail          = results["turbulence"]
        report.material_passed            = results["material"]["passed"]
        report.material_detail            = results["material"]

        report.overall_passed = all([
            report.governing_equations_passed,
            report.boundary_conditions_passed,
            report.conservation_passed,
            report.turbulence_passed if (pc and pc.turbulence_model) else True,
            report.material_passed   if (pc and "FEA" in pc.physics_type.value) else True,
        ])

        if not report.overall_passed:
            fixes = []
            if not report.governing_equations_passed:
                fixes.append(f"Increase lambda_continuity: continuity_error={results['gov_eq'].get('continuity_max','?')}")
            if not report.boundary_conditions_passed:
                fixes.append(f"Re-encode BC features: bc_error={results['bc'].get('no_slip_max','?')}")
            if not report.conservation_passed:
                fixes.append(f"Add conservation loss: mass_error={results['conservation'].get('mass_error','?')}")
            if not report.turbulence_passed:
                fixes.append(f"Add turbulence constraints: {results['turbulence'].get('failure_reason','?')}")
            report.fix_instructions = " | ".join(fixes)

            lambdas = {}
            if not report.governing_equations_passed:
                lambdas["continuity"] = state.physics_lambda_weights.get("continuity", 1.0) * 3.0
            if not report.boundary_conditions_passed:
                lambdas["bc"] = state.physics_lambda_weights.get("bc", 2.0) * 5.0
            report.physics_lambda_updates = lambdas

        state.physics_report = report
        state.physics_status = AgentStatus.PASSED if report.overall_passed else AgentStatus.FAILED

        if not report.boundary_conditions_passed:
            self._publish_insufficient_bc(state, results["bc"])

        logger.info(
            f"Physics: gov_eq={report.governing_equations_passed}, "
            f"bc={report.boundary_conditions_passed}, "
            f"conservation={report.conservation_passed}, "
            f"turbulence={report.turbulence_passed}, "
            f"overall={report.overall_passed}"
        )
        return state

    def _publish_insufficient_bc(self, state: AgentSystemState, bc_result: dict) -> None:
        """Notify dataset agent when BC data is missing."""
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
