"""
Master Orchestrator — builds and runs the full LangGraph pipeline.
Connects all agents in sequence with conditional routing.
"""

import uuid
from datetime import datetime
from langgraph.graph import StateGraph, END
from loguru import logger

from agents.orchestrator.agent_state import AgentSystemState, AgentStatus
from agents.data_agent.data_agent import DataAgent
from agents.analyst_agent.analyst_agent import AnalystAgent
from agents.selector_agent.deep_thinking_selector import DeepThinkingSelector
from agents.trainer_agent.trainer_agent import TrainerAgent
from agents.evaluator_agent.evaluator_agent import EvaluatorAgent
from agents.physics_agent.physics_master import PhysicsMasterAgent
from agents.iteration_agent.iteration_agent import IterationAgent
from agents.verifier_agent.verifier_agent import VerifierAgent
from agents.saver_agent.saver_agent import SaverAgent
from agents.self_learning.self_learning_updater import SelfLearningUpdater
from agents.self_learning.knowledge_base import KnowledgeBase
from memory.run_database import RunDatabase


class MasterOrchestrator:
    """
    Builds the complete LangGraph state machine.
    All agents connected with conditional routing.
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_agent = DataAgent(config)
        self.analyst_agent = AnalystAgent(config)
        self.selector_agent = DeepThinkingSelector(config)
        self.trainer_agent = TrainerAgent(config)
        self.evaluator_agent = EvaluatorAgent(config)
        self.physics_agent = PhysicsMasterAgent(config)
        self.iteration_agent = IterationAgent(config)
        self.verifier_agent = VerifierAgent(config)
        self.saver_agent = SaverAgent(config)

        # Self-learning components
        db_path = config.get("self_learning", {}).get("database_path", "memory/experience.db")
        self.db = RunDatabase(db_path=db_path)
        self.knowledge_base = KnowledgeBase(config, self.db)
        self.self_learning_updater = SelfLearningUpdater(config, self.db)

        # Dataset agent
        from agents.dataset_agent.dataset_orchestrator import DatasetOrchestrator
        self.dataset_orchestrator = DatasetOrchestrator(config)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentSystemState)

        # Register all agent nodes
        graph.add_node("data_agent",           self._run_data_agent)
        graph.add_node("dataset_orchestrator", self._run_dataset_orchestrator)
        graph.add_node("analyst_agent",        self._run_analyst_agent)
        graph.add_node("selector_agent",       self._run_selector_agent)
        graph.add_node("trainer_agent",        self._run_trainer_agent)
        graph.add_node("evaluator_agent",      self._run_evaluator_agent)
        graph.add_node("physics_agent",        self._run_physics_agent)
        graph.add_node("iteration_agent",      self._run_iteration_agent)
        graph.add_node("verifier_agent",       self._run_verifier_agent)
        graph.add_node("saver_agent",          self._run_saver_agent)
        graph.add_node("architect_agent",      self._run_architect_agent)
        graph.add_node("self_learning_update", self._run_self_learning_update)

        # Entry point
        graph.set_entry_point("data_agent")

        # data_agent -> dataset_orchestrator (if no data) OR -> analyst_agent
        graph.add_conditional_edges(
            "data_agent",
            self._route_after_data_agent,
            {"dataset_search": "dataset_orchestrator", "analyst": "analyst_agent"}
        )
        graph.add_edge("dataset_orchestrator", "analyst_agent")
        graph.add_edge("analyst_agent",  "selector_agent")
        graph.add_edge("selector_agent", "trainer_agent")
        graph.add_edge("trainer_agent",  "evaluator_agent")

        # Conditional: evaluator -> physics (if acc pass) or iteration (if fail)
        graph.add_conditional_edges(
            "evaluator_agent",
            self._route_after_evaluator,
            {"physics": "physics_agent", "iterate": "iteration_agent"}
        )

        # Conditional: physics -> verifier (if phys pass) or iteration (if fail)
        graph.add_conditional_edges(
            "physics_agent",
            self._route_after_physics,
            {"verify": "verifier_agent", "iterate": "iteration_agent"}
        )

        # Conditional: iteration -> trainer/selector/architect/end
        graph.add_conditional_edges(
            "iteration_agent",
            self._route_after_iteration,
            {
                "retry":      "trainer_agent",
                "select_new": "selector_agent",
                "architect":  "architect_agent",
                "end":        "self_learning_update",
            }
        )

        # Architect -> trainer
        graph.add_edge("architect_agent", "trainer_agent")

        # Conditional: verifier -> saver (pass) or iteration (fail)
        graph.add_conditional_edges(
            "verifier_agent",
            self._route_after_verifier,
            {"save": "saver_agent", "iterate": "iteration_agent"}
        )

        graph.add_edge("saver_agent", "self_learning_update")
        graph.add_edge("self_learning_update", END)

        return graph.compile()

    # ── Node wrappers ──────────────────────────────────────────
    def _run_data_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== DATA AGENT ===")
        return self.data_agent.run(state)

    def _run_analyst_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== ANALYST AGENT ===")
        return self.analyst_agent.run(state)

    def _run_selector_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== SELECTOR AGENT ===")
        return self.selector_agent.run(state)

    def _run_trainer_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info(f"=== TRAINER AGENT (attempt {state.current_attempt + 1}) ===")
        return self.trainer_agent.run(state)

    def _run_evaluator_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== EVALUATOR AGENT ===")
        return self.evaluator_agent.run(state)

    def _run_physics_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== PHYSICS MASTER AGENT ===")
        return self.physics_agent.run(state)

    def _run_iteration_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info(f"=== ITERATION AGENT (attempt {state.current_attempt}) ===")
        return self.iteration_agent.run(state)

    def _run_verifier_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== VERIFIER AGENT ===")
        return self.verifier_agent.run(state)

    def _run_saver_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== SAVER AGENT ===")
        return self.saver_agent.run(state)

    def _run_architect_agent(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== ARCHITECT AGENT ===")
        from agents.model_architect.architect_agent import ArchitectAgent
        architect = ArchitectAgent(self.config, self.db)
        state.architect_triggered = True
        return architect.run(state)

    def _run_dataset_orchestrator(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== DATASET ORCHESTRATOR ===")
        return self.dataset_orchestrator.run(state)

    def _run_self_learning_update(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== SELF-LEARNING UPDATE ===")
        return self.self_learning_updater.update(state)

    # ── Routing logic ──────────────────────────────────────────
    def _route_after_data_agent(self, state: AgentSystemState) -> str:
        if not state.data_path or state.search_datasets:
            logger.info("No data path or search_datasets=True — routing to Dataset Orchestrator")
            return "dataset_search"
        logger.info("Data path provided — skipping dataset search")
        return "analyst"

    def _route_after_evaluator(self, state: AgentSystemState) -> str:
        if state.evaluation_result and state.evaluation_result.passed:
            logger.info("Evaluator PASSED — routing to Physics Agent")
            return "physics"
        logger.info("Evaluator FAILED — routing to Iteration Agent")
        return "iterate"

    def _route_after_physics(self, state: AgentSystemState) -> str:
        if state.physics_report and state.physics_report.overall_passed:
            logger.info("Physics PASSED — routing to Verifier")
            return "verify"
        logger.info("Physics FAILED — routing to Iteration Agent")
        return "iterate"

    def _route_after_iteration(self, state: AgentSystemState) -> str:
        max_total = self.config.get("iteration", {}).get("total_max_attempts", 24)
        if state.current_attempt >= max_total:
            logger.error("Max attempts reached — pipeline ending without success")
            return "end"

        # Check message bus for REQUEST_MORE_DATA — re-run dataset search inline
        for msg in state.agent_messages:
            if not msg.get("handled") and msg.get("type") == "REQUEST_MORE_DATA":
                logger.info(f"Message bus: {msg['reason']} — re-running dataset orchestrator")
                state = self.dataset_orchestrator.run(state)
                msg["handled"] = True
                break

        # Trigger architect after 12 failed attempts if not already tried
        if (state.current_attempt >= 12 and
                not state.custom_model_attempted and
                state.problem_card is not None and
                self.knowledge_base.should_create_custom_model(
                    state.problem_card, state.current_attempt
                )):
            state.custom_model_attempted = True
            logger.info("Triggering Architect Agent after repeated failures")
            return "architect"

        fix = state.iteration_log[-1].fix_applied if state.iteration_log else ""
        if fix and "next_model" in fix:
            return "select_new"
        return "retry"

    def _route_after_verifier(self, state: AgentSystemState) -> str:
        if state.verification_passed:
            return "save"
        return "iterate"

    def run(self, data_path: str, software_source: str = "STAR-CCM+",
            search_datasets: bool = False) -> AgentSystemState:
        """Run the complete pipeline."""
        initial_state = AgentSystemState(
            data_path=data_path,
            software_source=software_source,
            search_datasets=search_datasets,
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        logger.info(f"Starting pipeline run: {initial_state.run_id}")
        logger.info(f"Knowledge base: {self.knowledge_base.get_system_stats()}")

        final_state = self.graph.invoke(initial_state)

        if final_state.pipeline_success:
            logger.success(f"Pipeline SUCCESS — model saved at {final_state.saved_model_path}")
        else:
            logger.error(f"Pipeline FAILED after {final_state.current_attempt} attempts")
        return final_state
