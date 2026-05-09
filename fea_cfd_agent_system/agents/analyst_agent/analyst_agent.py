"""
Analyst Agent — analyzes the FEA dataset and produces a Problem Card
that drives all downstream agents.
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
    Uses LLM + rule-based classification to analyze the dataset and produce
    a structured FEA Problem Card.

    Auto-classification rules (applied before LLM):
      - Time arrays in data     → loading_type = "dynamic"
      - Temperature field       → loading_type = "thermal"
      - Multiple load steps     → loading_type = "cyclic"
      - yield_stress in material → material_model = "elastoplastic"
      - 4 nodes/element         → element_type = "tet"
      - 8 nodes/element         → element_type = "hex"
      - 3 nodes/element         → element_type = "shell"
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm    = get_dev_llm(max_tokens=2000)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Analyst Agent: analyzing FEA dataset")
        state.analyst_status = AgentStatus.RUNNING

        try:
            schema = state.unified_schema
            # Apply rule-based classification first, then use LLM to refine
            rule_card = self._rule_classify(schema, state.software_source)
            try:
                problem_card = self._llm_refine(rule_card, schema, state.software_source)
            except Exception as e:
                logger.warning(f"LLM refinement failed: {e} — using rule-based card")
                problem_card = rule_card

            state.problem_card   = problem_card
            state.analyst_status = AgentStatus.PASSED

            logger.success(
                f"Analyst complete: {problem_card.physics_type.value}, "
                f"material={problem_card.material_model}, "
                f"loading={problem_card.loading_type}, "
                f"solver={problem_card.solver_source}"
            )
            state.thinking_log.append(
                f"ANALYST: {problem_card.physics_type.value} | "
                f"element={problem_card.element_type} | "
                f"n={problem_card.data_size}"
            )

        except Exception as e:
            state.analyst_status = AgentStatus.FAILED
            state.error_message  = f"Analyst Agent failed: {str(e)}"
            logger.error(state.error_message)
            state.problem_card   = self._rule_classify(state.unified_schema, state.software_source)
            state.analyst_status = AgentStatus.PASSED
            logger.warning("Analyst: using rule-based fallback problem card")

        return state

    def _rule_classify(self, schema: dict, software: str) -> ProblemCard:
        """Rule-based FEA auto-classification from schema fields."""
        fields        = schema.get("fields", {})
        fields_lower  = {k.lower(): k for k in (fields if isinstance(fields, dict) else {})}
        cases         = schema.get("cases", [])
        first_case    = cases[0] if cases else {}

        # --- Physics type ---
        has_temp   = any(f in fields_lower for f in ["temperature", "temp", "t", "nt11"])
        has_time   = any(f in fields_lower for f in ["time", "timestep", "t_step"])
        n_steps    = schema.get("load_steps", first_case.get("load_steps", 1))

        if has_temp and any(f in fields_lower for f in ["stress", "displacement"]):
            physics = PhysicsType.THERMAL_STRUCTURAL
            loading = "thermal"
        elif has_temp:
            physics = PhysicsType.THERMAL
            loading = "thermal"
        elif has_time or (isinstance(n_steps, int) and n_steps > 10):
            physics = PhysicsType.FEA_DYNAMIC
            loading = "dynamic"
        elif any(f in fields_lower for f in ["plastic_strain", "peeq", "yield"]):
            physics = PhysicsType.FEA_STATIC_NONLINEAR
            loading = "static"
        else:
            physics = PhysicsType.FEA_STATIC_LINEAR
            loading = "static"

        # --- Material model ---
        mat = first_case.get("material_properties", {})
        if mat.get("yield_strength") or mat.get("yield_stress"):
            material_model = "elastoplastic"
        elif mat.get("shear_modulus") or mat.get("mu"):
            material_model = "hyperelastic"
        else:
            material_model = "linear_elastic"

        # --- Element type ---
        elems = first_case.get("elements")
        if elems is not None and hasattr(elems, "shape") and len(elems) > 0:
            k = elems.shape[-1] if elems.ndim > 1 else 0
            element_type = {4: "tet", 8: "hex", 3: "shell", 2: "beam"}.get(k, "mixed")
        else:
            element_type = "tet"

        # --- Mesh type ---
        mesh_type_str = schema.get("mesh_type", "")
        if "hex" in mesh_type_str:
            mesh_type = MeshType.UNSTRUCTURED_HEX
        elif "shell" in element_type or "beam" in element_type:
            mesh_type = MeshType.UNSTRUCTURED_TET
        else:
            mesh_type = MeshType.UNSTRUCTURED_TET

        # --- Accuracy thresholds by physics type ---
        acc = self._accuracy_thresholds(physics)

        return ProblemCard(
            problem_type        = ProblemType.FIELD_REGRESSION,
            physics_type        = physics,
            mesh_type           = mesh_type,
            input_features      = self._infer_inputs(fields_lower),
            output_targets      = self._infer_outputs(fields_lower),
            data_size           = schema.get("n_cases", 0),
            n_nodes             = schema.get("n_nodes", 0),
            n_cells             = schema.get("n_elements", schema.get("n_cells", 0)),
            physics_constraints = self._infer_constraints(physics),
            material_model      = material_model,
            loading_type        = loading,
            element_type        = element_type,
            n_load_steps        = int(n_steps) if isinstance(n_steps, (int, float)) else 1,
            solver_source       = software,
            accuracy_threshold  = acc,
        )

    def _llm_refine(self, base_card: ProblemCard, schema: dict, software: str) -> ProblemCard:
        """Ask LLM to refine the rule-based classification."""
        summary = {
            "fields":         list(schema.get("fields", {}).keys())[:20],
            "n_cases":        schema.get("n_cases", 0),
            "n_nodes":        schema.get("n_nodes", 0),
            "solver_source":  software,
            "mesh_type":      schema.get("mesh_type", ""),
        }
        prompt = f"""
You are an expert FEA ML engineer. A rule-based classifier produced this initial FEA problem card.
Verify and refine it based on the dataset summary below.

Dataset summary:
{json.dumps(summary, indent=2)}

Initial classification:
  physics_type: {base_card.physics_type.value}
  material_model: {base_card.material_model}
  loading_type: {base_card.loading_type}
  element_type: {base_card.element_type}

Output ONLY valid JSON with these keys (keep existing values if confident they are correct):
{{
  "physics_type": "FEA_static_linear",
  "material_model": "linear_elastic",
  "loading_type": "static",
  "element_type": "tet",
  "geometry_description": "plate with hole",
  "special_flags": []
}}

physics_type must be one of: FEA_static_linear, FEA_static_nonlinear, FEA_dynamic,
  thermal, thermal_structural, multiphysics
material_model: linear_elastic | hyperelastic | elastoplastic
loading_type: static | dynamic | thermal | cyclic
element_type: tet | hex | shell | beam | mixed

Output ONLY JSON. No explanation.
"""
        response = self.llm.invoke(prompt)
        content  = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)

        # Update base card with LLM refinements
        try:
            base_card.physics_type  = PhysicsType(data.get("physics_type", base_card.physics_type.value))
        except ValueError:
            pass
        base_card.material_model      = data.get("material_model", base_card.material_model)
        base_card.loading_type        = data.get("loading_type",   base_card.loading_type)
        base_card.element_type        = data.get("element_type",   base_card.element_type)
        base_card.geometry_description = data.get("geometry_description", "")
        base_card.special_flags       = data.get("special_flags", [])
        base_card.accuracy_threshold  = self._accuracy_thresholds(base_card.physics_type)
        return base_card

    def _accuracy_thresholds(self, physics: PhysicsType) -> dict:
        if physics == PhysicsType.FEA_STATIC_LINEAR:
            return {"r2": 0.95, "rel_l2": 0.03, "max_point_error": 0.10}
        if physics == PhysicsType.FEA_STATIC_NONLINEAR:
            return {"r2": 0.93, "rel_l2": 0.05, "max_point_error": 0.12}
        if physics == PhysicsType.FEA_DYNAMIC:
            return {"r2": 0.90, "rel_l2": 0.08, "max_point_error": 0.15}
        if physics in (PhysicsType.THERMAL, PhysicsType.THERMAL_STRUCTURAL):
            return {"r2": 0.95, "rel_l2": 0.03, "max_point_error": 0.10}
        return {"r2": 0.92, "rel_l2": 0.05, "max_point_error": 0.12}

    def _infer_inputs(self, fields_lower: dict) -> list:
        inputs = ["geometry", "material_props", "boundary_conditions"]
        if "temperature" in fields_lower:
            inputs.append("temperature_distribution")
        return inputs

    def _infer_outputs(self, fields_lower: dict) -> list:
        targets = []
        for std, keys in [
            ("displacement_field", ["displacement", "disp", "u"]),
            ("stress_field",       ["stress", "s", "sigma"]),
            ("von_mises_field",    ["von_mises", "mises", "seqv"]),
            ("strain_field",       ["strain", "e", "epsilon"]),
            ("temperature_field",  ["temperature", "temp"]),
        ]:
            if any(k in fields_lower for k in keys):
                targets.append(std)
        return targets or ["displacement_field", "stress_field"]

    def _infer_constraints(self, physics: PhysicsType) -> list:
        base = ["equilibrium", "strain_compatibility"]
        if physics == PhysicsType.FEA_DYNAMIC:
            base.extend(["momentum_balance", "damping"])
        if physics in (PhysicsType.THERMAL, PhysicsType.THERMAL_STRUCTURAL):
            base.extend(["energy_balance", "fouriers_law"])
        return base
