"""Discovery Agent — finds new ML models from arXiv and adds them to the registry."""

from loguru import logger
from memory.run_database import RunDatabase
from agents.self_learning.discovery_agent import DiscoveryAgent as SelfLearningDiscovery


class DiscoveryAgent:
    """Thin wrapper that delegates to the self-learning DiscoveryAgent."""

    def __init__(self, db: RunDatabase):
        self.agent = SelfLearningDiscovery(db)

    def run_discovery(self, max_papers: int = 30) -> list:
        """Run arXiv discovery and return new model candidates."""
        return self.agent.run_discovery(max_papers=max_papers)
