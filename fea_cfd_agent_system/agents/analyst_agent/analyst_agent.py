"""
Analyst Agent — thinks deeply about the dataset and produces
a Problem Card that drives all downstream agents.
"""

import json
from loguru import logger
from agents.shared.llm_factory import get_dev_llm
from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, ProblemCard,
    PhysicsType, MeshType, ProblemType
)


class AnalystAgent:
    """
    Uses LLM to analyze dataset and produce a structured Problem Card.
    Performs 5-step thinking:
    1. Data profiling
    2. Problem classification
    3. Data size decision
    4. Physics identification
    5. Problem Card generation
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm = get_dev_llm(max_tokens=2000)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Analyst Agent: analyzing dataset")
        state.analyst_status = AgentStatus.RUNNING

        try:
            schema = state.unified_schema
            problem_card = self._think_and_classify(schema, state.software_source)
            state.problem_card = problem_card
            state.analyst_status = AgentStatus.PASSED

            logger.success(
                f"Analyst complete: {problem_card.physics_type.value}, "
                f"{problem_card.mesh_type.value}, "
                f"{problem_card.data_size} cases"
            )
            state.thinking_log.append(
                f"ANALYST: {problem_card.physics_type.value} | "
                f"mesh={problem_card.mesh_type.value} | "
                f"n={problem_card.data_size}"
            )

        except Exception as e:
            state.analyst_status = AgentStatus.FAILED
            state.error_message = f"Analyst Agent failed: {str(e)}"
            logger.error(state.error_message)
            # Fallback: create a default problem card based on available info
            state.problem_card = self._fallback_problem_card(state.unified_schema, state.software_source)
            state.analyst_status = AgentStatus.PASSED
            logger.warning("Analyst: using fallback problem card")

        return state

    def _think_and_classify(self, schema: dict, software: str) -> ProblemCard:
        """Ask LLM to classify the problem from dataset schema."""
        prompt = f"""
You are an expert in computational fluid dynamics (CFD) and finite element analysis (FEA).
Analyze this simulation dataset schema and classify the problem.

Dataset Schema:
{json.dumps(schema, indent=2, default=str)}

Software Source: {software}

Based on the fields present (velocity u/v/w = CFD, stress/strain = FEA, etc.),
classify this problem and output ONLY valid JSON with this exact structure:

{{
  "problem_type": "field_regression",
  "physics_type": "CFD_incompressible_turbulent",
  "mesh_type": "unstructured_polyhedral",
  "input_features": ["geometry", "inlet_velocity", "Re"],
  "output_targets": ["velocity_field", "pressure_field"],
  "physics_constraints": ["div_u_equals_0", "no_slip", "mass_conservation"],
  "turbulence_model": "k-omega_SST",
  "re_number": 85000,
  "geometry_description": "rectangular duct with curved corners",
  "special_flags": ["curved_boundaries_present", "boundary_layer_critical"]
}}

physics_type must be one of:
CFD_incompressible, CFD_incompressible_turbulent, CFD_compressible,
FEA_static_linear, FEA_static_nonlinear, FEA_dynamic, thermal,
thermal_structural, multiphysics

mesh_type must be one of:
structured, unstructured_polyhedral, unstructured_tetrahedral,
unstructured_hexahedral, point_cloud, tabular

Output ONLY the JSON. No preamble. No explanation.
"""
        response = self.llm.invoke(prompt)
        content = response.content.strip()
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)

        return ProblemCard(
            problem_type=ProblemType(data.get("problem_type", "field_regression")),
            physics_type=PhysicsType(data.get("physics_type", "CFD_incompressible_turbulent")),
            mesh_type=MeshType(data.get("mesh_type", "unstructured_polyhedral")),
            input_features=data.get("input_features", []),
            output_targets=data.get("output_targets", []),
            data_size=schema.get("n_cases", 0),
            n_nodes=schema.get("n_nodes", 0),
            n_cells=schema.get("n_cells", 0),
            physics_constraints=data.get("physics_constraints", []),
            turbulence_model=data.get("turbulence_model"),
            re_number=data.get("re_number"),
            accuracy_threshold={"r2": 0.92, "rel_l2": 0.05, "max_point_error": 0.15},
            special_flags=data.get("special_flags", []),
            software_source=software,
            geometry_description=data.get("geometry_description", ""),
        )

    def _fallback_problem_card(self, schema: dict, software: str) -> ProblemCard:
        """Rule-based fallback when LLM fails."""
        fields = schema.get("fields", [])
        fields_lower = [f.lower() for f in fields]

        if any(f in fields_lower for f in ["velocity", "pressure", "k", "omega"]):
            physics = PhysicsType.CFD_INCOMPRESSIBLE_TURBULENT
            turbulence = "k-omega_SST"
        elif any(f in fields_lower for f in ["stress", "displacement", "strain"]):
            physics = PhysicsType.FEA_STATIC_LINEAR
            turbulence = None
        else:
            physics = PhysicsType.CFD_INCOMPRESSIBLE_TURBULENT
            turbulence = None

        return ProblemCard(
            physics_type=physics,
            mesh_type=MeshType.UNSTRUCTURED_POLY,
            data_size=schema.get("n_cases", 0),
            n_nodes=schema.get("n_nodes", 0),
            n_cells=schema.get("n_cells", 0),
            turbulence_model=turbulence,
            software_source=software,
        )
