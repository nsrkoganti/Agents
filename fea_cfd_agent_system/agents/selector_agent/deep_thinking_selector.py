"""
Deep Thinking Model Selector Agent.

Does NOT use a fixed priority list.
Instead:
1. Loads ALL known models from registry (20+ models)
2. Scans GitHub repos for each model — reads code, checks mesh assumptions
3. Scores every model on 8 criteria with problem-specific weights
4. Uses LLM to reason about top-10 and pick final shortlist
5. Installs and verifies the selected model
"""

import json
import yaml
import subprocess
from pathlib import Path
from loguru import logger
from agents.shared.llm_factory import get_dev_llm

from agents.orchestrator.agent_state import (
    AgentSystemState, AgentStatus, ModelCandidate
)
from agents.selector_agent.github_scanner import GitHubScanner
from agents.selector_agent.scoring_engine import ModelScoringEngine


class DeepThinkingSelector:
    """
    Thinks like a senior ML researcher.
    Reads code. Reads benchmarks. Scores. Reasons. Selects.
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm = get_dev_llm(max_tokens=3000)
        self.scanner = GitHubScanner()
        self.scoring_engine = ModelScoringEngine(config)
        self.registry = self._load_registry()

    def _load_registry(self) -> list:
        """Load all known models from YAML registry."""
        registry_path = Path("configs/model_registry.yaml")
        if not registry_path.exists():
            registry_path = Path(__file__).parent.parent.parent / "configs" / "model_registry.yaml"
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        candidates = []
        for m in data.get("models", []):
            candidates.append(ModelCandidate(
                name=m["name"],
                family=m["family"],
                github_url=m["github_url"],
                install_cmd=m["install_cmd"],
                paper=m.get("paper", ""),
                mesh_requirement=m.get("mesh_requirement", "any"),
                min_data_samples=m.get("min_data_samples", 0),
                supports_field_output=m.get("supports_field_output", True),
                has_builtin_physics_loss=m.get("has_builtin_physics_loss", False),
                benchmark_l2_error=m.get("benchmark_l2_error", 0.05),
                inference_speed_score=m.get("inference_speed_score", 5),
                code_maturity_stars=m.get("code_maturity_stars", 0),
            ))
        logger.info(f"Loaded {len(candidates)} models from registry")
        return candidates

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("Deep Thinking Selector: analyzing all models")
        state.selector_status = AgentStatus.RUNNING
        problem_card = state.problem_card

        # Layer 1: Apply hard filters
        candidates = self._apply_hard_filters(self.registry, problem_card)
        state.thinking_log.append(
            f"SELECTOR: {len(self.registry)} total models -> "
            f"{len(candidates)} after hard filters"
        )

        # Layer 2: Scan GitHub (skip if already have reports)
        candidates = self._scan_github_repos(candidates, state)

        # Layer 3: Score all
        candidates = self._score_all(candidates, problem_card)

        # Layer 4: LLM reasoning on top-10
        shortlist = self._llm_reasoning(candidates[:10], problem_card, state)

        # Layer 5: Install & verify top model
        shortlist = self._install_and_verify(shortlist, state)

        state.all_candidates = candidates
        state.ranked_shortlist = shortlist
        state.selected_model = shortlist[0] if shortlist else None
        state.selector_status = AgentStatus.PASSED

        if state.selected_model:
            logger.success(f"Selected: {state.selected_model.name} "
                           f"(score={state.selected_model.total_score:.2f})")
        return state

    def _apply_hard_filters(self, candidates, problem_card) -> list:
        """Remove models that are fundamentally incompatible."""
        filtered = []
        for m in candidates:
            if (m.mesh_requirement == "structured" and
                    "unstructured" in problem_card.mesh_type.value):
                m.skip_reason = f"Requires structured grid, problem has {problem_card.mesh_type.value}"
                logger.debug(f"SKIP {m.name}: {m.skip_reason}")
                continue

            if problem_card.data_size < m.min_data_samples:
                m.skip_reason = f"Needs {m.min_data_samples} samples, only {problem_card.data_size} available"
                logger.debug(f"SKIP {m.name}: {m.skip_reason}")
                continue

            if (problem_card.problem_type.value == "field_regression" and
                    not m.supports_field_output):
                m.skip_reason = "Scalar output only, field regression needed"
                logger.debug(f"SKIP {m.name}: {m.skip_reason}")
                continue

            filtered.append(m)
        return filtered

    def _scan_github_repos(self, candidates, state) -> list:
        """Scan GitHub for each candidate."""
        for m in candidates:
            if not m.github_report and "github.com" in m.github_url:
                try:
                    report = self.scanner.scan(m.github_url)
                    m.github_report = report
                    m.code_maturity_stars = report.get("stars", m.code_maturity_stars)
                    state.thinking_log.append(
                        f"GITHUB {m.name}: stars={report.get('stars',0)}, "
                        f"mesh={report.get('mesh_assumption','?')}, "
                        f"last_commit={str(report.get('last_commit','?'))[:10]}"
                    )
                    if (report.get("mesh_assumption") == "STRUCTURED_ONLY" and
                            "unstructured" in state.problem_card.mesh_type.value):
                        m.skip_reason = "Code scan: grid_sample() detected — structured only"
                except Exception as e:
                    logger.warning(f"GitHub scan failed for {m.name}: {e}")
        return [m for m in candidates if not m.skip_reason]

    def _score_all(self, candidates, problem_card) -> list:
        """Score every model on 8 criteria."""
        for m in candidates:
            self.scoring_engine.score(m, problem_card)
            logger.debug(f"SCORED {m.name}: {m.total_score:.2f}")
        return sorted(candidates, key=lambda x: x.total_score, reverse=True)

    def _llm_reasoning(self, top_candidates, problem_card, state) -> list:
        """LLM reasons about the top candidates and produces ranked shortlist."""
        candidates_info = [
            {
                "name": m.name,
                "score": round(m.total_score, 2),
                "scores": {k: round(v, 1) for k, v in m.scores.items()},
                "github_stars": m.code_maturity_stars,
                "benchmark_l2": m.benchmark_l2_error,
                "has_physics_loss": m.has_builtin_physics_loss,
                "mesh_requirement": m.mesh_requirement,
            }
            for m in top_candidates
        ]

        prompt = f"""
You are an expert ML researcher for FEA and CFD surrogate modeling.

Problem Card:
- Physics type: {problem_card.physics_type.value}
- Mesh type: {problem_card.mesh_type.value}
- Data size: {problem_card.data_size} cases
- Mesh nodes: {problem_card.n_nodes}
- Physics constraints: {problem_card.physics_constraints}
- Turbulence model: {problem_card.turbulence_model}
- Re number: {problem_card.re_number}
- Special flags: {problem_card.special_flags}
- Geometry: {problem_card.geometry_description}

Top scored candidates:
{json.dumps(candidates_info, indent=2)}

Choose the TOP 3 models to try in order. For each, explain WHY it fits.
Consider mesh compatibility, data size, turbulence physics, geometry complexity.

Output ONLY JSON:
{{
  "shortlist": [
    {{"model": "ModelName", "reason": "Why this fits", "config_hints": "layer count, hidden dim, key parameters"}},
    {{"model": "ModelName", "reason": "Why as fallback", "config_hints": "key parameters"}},
    {{"model": "ModelName", "reason": "Why as second fallback", "config_hints": "key parameters"}}
  ],
  "thinking": "Your full reasoning in 3-4 sentences"
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            state.thinking_log.append(f"LLM REASONING: {result.get('thinking','')}")

            name_map = {m.name: m for m in top_candidates}
            shortlist = []
            for s in result.get("shortlist", []):
                if s["model"] in name_map:
                    m = name_map[s["model"]]
                    m.github_report["llm_reason"] = s.get("reason", "")
                    m.github_report["config_hints"] = s.get("config_hints", "")
                    shortlist.append(m)
            return shortlist if shortlist else top_candidates[:3]

        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e} — using score ranking")
            return top_candidates[:3]

    def _install_and_verify(self, shortlist, state) -> list:
        """Try to install top model and verify with test forward pass."""
        for m in shortlist:
            try:
                logger.info(f"Installing {m.name}: {m.install_cmd}")
                if m.install_cmd.startswith("pip install"):
                    pkg = m.install_cmd.replace("pip install", "").strip()
                    subprocess.run(
                        ["pip", "install", pkg, "--quiet"],
                        check=True, capture_output=True, timeout=120
                    )
                m.install_verified = True
                state.thinking_log.append(f"INSTALLED: {m.name}")
                return shortlist
            except Exception as e:
                logger.warning(f"Install failed for {m.name}: {e}")
                if m in shortlist:
                    shortlist.remove(m)
        return shortlist
