"""
Dataset Orchestrator — coordinates search → select → download → validate.
Plugs into LangGraph as a single node. Updates AgentSystemState with
discovered_datasets, selected_dataset, dataset_download_path.
"""

import datetime
from loguru import logger
from agents.orchestrator.agent_state import AgentSystemState, AgentStatus
from agents.dataset_agent.dataset_searcher import DatasetSearchAgent
from agents.dataset_agent.dataset_downloader import DatasetDownloadAgent
from agents.dataset_agent.dataset_validator import DatasetValidatorAgent


class DatasetOrchestrator:
    """
    Full dataset acquisition pipeline:
    1. Search HuggingFace / GitHub / Zenodo
    2. LLM ranks candidates
    3. Download best candidate
    4. Validate quality
    5. Update state.data_path so DataAgent can load it normally
    """

    def __init__(self, config: dict):
        self.config    = config
        self.searcher  = DatasetSearchAgent(config)
        self.downloader= DatasetDownloadAgent(config)
        self.validator = DatasetValidatorAgent()

    def run(self, state: AgentSystemState) -> AgentSystemState:
        logger.info("=== DATASET ORCHESTRATOR ===")
        state.dataset_agent_status = AgentStatus.RUNNING

        # Build search context from problem_card or raw state
        physics_type = "cfd_incompressible"
        mesh_type    = "unstructured"
        description  = ""

        if state.problem_card:
            physics_type = state.problem_card.physics_type.value
            mesh_type    = state.problem_card.mesh_type.value
            description  = (f"Re={state.problem_card.re_number}, "
                            f"turbulence={state.problem_card.turbulence_model}")
        elif state.unified_schema:
            physics_type = state.unified_schema.get("physics_type", physics_type)
            mesh_type    = state.unified_schema.get("mesh_type",    mesh_type)

        min_samples = max(100, state.problem_card.data_size // 2
                         if state.problem_card else 100)

        # 1. Search
        datasets = self.searcher.search(
            physics_type=physics_type,
            mesh_type=mesh_type,
            min_samples=min_samples,
            problem_description=description,
        )
        state.discovered_datasets = datasets
        state.thinking_log.append(
            f"DATASET SEARCH: found {len(datasets)} candidate(s) for {physics_type}"
        )

        if not datasets:
            logger.warning("Dataset search returned no results")
            state.dataset_agent_status = AgentStatus.FAILED
            self._publish_message(state, "dataset_agent", "orchestrator",
                                  "NO_DATASETS_FOUND",
                                  f"Search for {physics_type} returned 0 results")
            return state

        # 2. Try candidates in ranked order until one validates
        for candidate in datasets[:5]:
            logger.info(f"Trying dataset: {candidate.get('name')} "
                        f"({candidate.get('source')})")

            # Check cache first
            local_path = self.downloader.get_cache_path(candidate)
            if not local_path:
                local_path = self.downloader.download(candidate)

            if not local_path:
                logger.warning(f"Download failed for {candidate.get('name')}")
                continue

            # 3. Validate
            report = self.validator.validate(local_path, candidate)
            state.dataset_quality_report = report

            if report["valid"]:
                state.selected_dataset      = candidate
                state.dataset_download_path = local_path
                state.data_path             = local_path  # feed into DataAgent
                state.dataset_agent_status  = AgentStatus.PASSED
                state.thinking_log.append(
                    f"DATASET: selected '{candidate.get('name')}' "
                    f"({report['format']}, {report['n_files']} files)"
                )
                self._publish_message(
                    state, "dataset_agent", "data_agent",
                    "DATASET_READY",
                    f"Downloaded and validated: {candidate.get('name')} at {local_path}",
                )
                logger.success(
                    f"Dataset ready: {candidate.get('name')} → {local_path}"
                )
                return state

            logger.warning(
                f"Dataset '{candidate.get('name')}' failed validation: "
                f"{report['issues']}"
            )

        # All candidates failed
        logger.error("All dataset candidates failed validation")
        state.dataset_agent_status = AgentStatus.FAILED
        self._publish_message(state, "dataset_agent", "orchestrator",
                              "ALL_DATASETS_INVALID",
                              "All downloaded datasets failed quality checks")
        return state

    def _publish_message(self, state: AgentSystemState,
                          from_agent: str, to_agent: str,
                          msg_type: str, reason: str):
        state.agent_messages.append({
            "from":      from_agent,
            "to":        to_agent,
            "type":      msg_type,
            "reason":    reason,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "handled":   False,
        })
