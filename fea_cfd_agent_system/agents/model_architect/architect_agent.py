"""
Architect Agent — the master designer.

When all known models fail, this agent:
1. Analyzes WHY they failed
2. Decides what architectural properties the new model needs
3. Designs the ArchitectureDNA
4. Generates Python code for the model
5. Validates the code runs
6. Passes the custom model to the Trainer Agent
7. If it succeeds, saves it to the database for future use
8. If it fails, mutates the DNA and tries again
"""

import json
import uuid
import torch
import datetime
from typing import Optional, Tuple
from loguru import logger
from langchain_anthropic import ChatAnthropic

from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, ModelCandidate
)
from agents.model_architect.architecture_dna import (
    ArchitectureDNA, pinn_dna, transolver_dna, fno_dna,
    gnn_dna, hybrid_transolver_pinn_dna
)
from agents.model_architect.code_generator import CodeGenerator
from agents.model_architect.nas_engine import NASEngine
from memory.run_database import RunDatabase


class ArchitectAgent:
    """
    Designs and builds custom neural architectures.
    Called when the existing model library is exhausted.
    """

    MAX_DESIGN_ATTEMPTS = 5
    MAX_CODE_RETRIES    = 3

    def __init__(self, config: dict, db: RunDatabase):
        self.config    = config
        self.db        = db
        self.llm       = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=3000)
        self.generator = CodeGenerator()
        self.nas       = NASEngine(config)

    def run(self, state: AgentSystemState) -> AgentSystemState:
        """Design a custom model for this problem."""
        logger.info("ARCHITECT AGENT: Designing custom model")

        problem_card   = state.problem_card
        iteration_log  = state.iteration_log
        physics_report = state.physics_report

        design_requirements = self._analyze_failures(
            iteration_log, physics_report, problem_card
        )
        state.thinking_log.append(
            f"ARCHITECT: {design_requirements.get('summary', 'analyzing...')}"
        )

        dna = self._choose_template(design_requirements, problem_card)
        state.thinking_log.append(
            f"ARCHITECT: Designing {dna.name} (family={dna.family})"
        )

        if design_requirements.get("use_nas", False):
            dna = self.nas.refine_dna(dna, problem_card, n_trials=20)

        custom_model, dna, code = self._generate_and_validate(dna, problem_card)

        if custom_model is None:
            logger.error("Architect: all design attempts failed")
            return state

        model_id = f"custom_{dna.name}_{uuid.uuid4().hex[:8]}"
        self.db.save_custom_model(
            model_id=model_id,
            name=dna.name,
            dna=dna.to_dict(),
            code=code,
            problem=problem_card.physics_type.value,
            r2=0.0,
            generation=dna.generation,
        )

        state.custom_model_dna  = dna.to_dict()
        state.custom_model_code = code

        candidate = ModelCandidate(
            name=dna.name,
            family=f"custom_{dna.family}",
            github_url="local://custom_models",
            install_cmd="already_built",
            mesh_requirement=dna.mesh_type,
            supports_field_output=True,
            has_builtin_physics_loss=dna.has_physics_loss,
            github_report={
                "custom_model": True,
                "model_object": custom_model,
                "model_code":   code,
                "dna":          dna.to_dict(),
                "model_id":     model_id,
            }
        )

        state.ranked_shortlist.insert(0, candidate)
        state.selected_model = candidate

        logger.success(
            f"Architect: designed and validated {dna.name} "
            f"(family={dna.family}, physics_loss={dna.has_physics_loss})"
        )
        return state

    def _analyze_failures(self, iteration_log, physics_report,
                            problem_card) -> dict:
        """Use LLM to understand WHY all existing models failed."""
        failures_summary = []
        for record in (iteration_log or [])[-8:]:
            failures_summary.append({
                "model":     record.model_name,
                "r2":        record.evaluation_result.r2_score if record.evaluation_result else 0,
                "failure":   record.failure_reason,
                "fix_tried": record.fix_applied,
            })

        physics_failures = {}
        if physics_report:
            physics_failures = {
                "governing_eq": physics_report.governing_equations_passed,
                "bc":           physics_report.boundary_conditions_passed,
                "conservation": physics_report.conservation_passed,
                "turbulence":   physics_report.turbulence_passed,
            }

        prompt = f"""
You are an expert neural architecture designer for CFD/FEA simulation.
All standard models have failed. Analyze why and specify what a new model needs.

Problem:
- Physics type: {problem_card.physics_type.value if problem_card else 'unknown'}
- Mesh type:    {problem_card.mesh_type.value if problem_card else 'unknown'}
- Data size:    {problem_card.data_size if problem_card else 0}
- Re number:    {problem_card.re_number if problem_card else None}
- Flags:        {problem_card.special_flags if problem_card else []}
- Turbulence:   {problem_card.turbulence_model if problem_card else None}

Failed attempts:
{json.dumps(failures_summary, indent=2)}

Physics failures:
{json.dumps(physics_failures, indent=2)}

Output ONLY JSON:
{{
  "summary": "one sentence: why models are failing and what's needed",
  "root_cause": "underfitting | overfitting | mesh_incompatibility | physics_violation | geometry_complexity",
  "needs_physics_loss": true,
  "needs_attention_over_physics_states": true,
  "needs_graph_structure": false,
  "needs_fourier_basis": false,
  "preferred_family": "hybrid | transformer | pinn | gnn | operator",
  "key_requirement": "one sentence",
  "use_nas": false,
  "recommended_hidden_dim": 256,
  "recommended_n_layers": 8,
  "recommended_n_slices": 32
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content  = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Failure analysis LLM call failed: {e}")
            return {
                "summary":       "Standard models failed — trying hybrid approach",
                "preferred_family": "hybrid",
                "needs_physics_loss": True,
                "needs_attention_over_physics_states": True,
                "use_nas":       False,
                "recommended_hidden_dim": 128,
                "recommended_n_layers":   6,
                "recommended_n_slices":   16,
            }

    def _choose_template(self, requirements: dict,
                          problem_card) -> ArchitectureDNA:
        family   = requirements.get("preferred_family", "hybrid")
        dim      = requirements.get("recommended_hidden_dim", 128)
        n_layers = requirements.get("recommended_n_layers", 6)
        n_slices = requirements.get("recommended_n_slices", 16)

        if family == "pinn":
            return pinn_dna(hidden_dim=dim, n_layers=n_layers)
        elif family == "transformer" and requirements.get("needs_attention_over_physics_states"):
            return transolver_dna(hidden_dim=dim, n_layers=n_layers, n_slices=n_slices)
        elif family == "gnn":
            return gnn_dna(hidden_dim=dim, n_layers=n_layers)
        elif family == "operator" and not requirements.get("needs_graph_structure"):
            return fno_dna(hidden_dim=dim, n_layers=n_layers)
        else:
            return hybrid_transolver_pinn_dna(
                hidden_dim=dim, n_layers=n_layers, n_slices=n_slices
            )

    def _generate_and_validate(
        self, dna: ArchitectureDNA, problem_card
    ) -> Tuple[Optional[torch.nn.Module], ArchitectureDNA, str]:
        pc_dict = {
            "physics_type":   problem_card.physics_type.value if problem_card else "",
            "mesh_type":      problem_card.mesh_type.value if problem_card else "",
            "data_size":      problem_card.data_size if problem_card else 0,
            "n_nodes":        problem_card.n_nodes if problem_card else 0,
            "re_number":      problem_card.re_number if problem_card else None,
            "output_targets": problem_card.output_targets if problem_card else [],
        }

        for attempt in range(self.MAX_CODE_RETRIES):
            logger.info(f"Code generation attempt {attempt + 1}/{self.MAX_CODE_RETRIES}")
            code = self.generator.generate(dna, pc_dict)

            output_dim = len(problem_card.output_targets) if problem_card and problem_card.output_targets else 4
            is_valid, error = self.generator.validate(code, 3, output_dim)

            if is_valid:
                model = self._instantiate_from_code(code, dna.name, 3, output_dim)
                if model is not None:
                    return model, dna, code
            else:
                logger.warning(f"Generated code invalid (attempt {attempt + 1}): {error}")
                if dna.core_blocks:
                    dna.core_blocks = dna.core_blocks[:-1]

        return None, dna, ""

    def _instantiate_from_code(self, code: str, class_name: str,
                                 input_dim: int, output_dim: int):
        """Dynamically load and instantiate the generated model."""
        import tempfile, importlib.util
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name

        try:
            spec   = importlib.util.spec_from_file_location("custom", tmp)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                        issubclass(attr, torch.nn.Module) and
                        attr is not torch.nn.Module):
                    return attr(input_dim=input_dim, output_dim=output_dim)
        except Exception as e:
            logger.error(f"Model instantiation failed: {e}")
        return None
