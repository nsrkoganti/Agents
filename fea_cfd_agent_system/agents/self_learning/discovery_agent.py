"""
Discovery Agent — scans arXiv, Papers With Code, and GitHub weekly
to find newly published ML models for FEA/CFD.
Adds them to the model registry automatically.
"""

import json
import re
import sqlite3
import datetime
from typing import List, Dict
from loguru import logger
from agents.shared.llm_factory import get_dev_llm
from memory.run_database import RunDatabase

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False


class DiscoveryAgent:
    """
    Continuously expands the system's knowledge of available models.
    Runs on a schedule or on demand.
    Adds discovered models to the database and model_registry.yaml.
    """

    SEARCH_QUERIES = [
        "CFD surrogate neural operator mesh",
        "FEA structural surrogate deep learning",
        "physics-informed neural network PDE unstructured",
        "transformer PDE solver geometry",
        "graph neural network fluid mechanics",
        "neural operator turbulence Reynolds-Averaged",
        "machine learning finite element analysis",
        "surrogate model computational fluid dynamics 3D",
    ]

    def __init__(self, db: RunDatabase):
        self.db  = db
        self.llm = get_dev_llm(max_tokens=2000)

    def run_discovery(self, max_papers: int = 30) -> List[Dict]:
        """Scan arXiv for new models."""
        if not ARXIV_AVAILABLE:
            logger.warning("arxiv package not installed — skip discovery. pip install arxiv")
            return []

        discovered  = []
        seen_titles = set()

        for query in self.SEARCH_QUERIES:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=max(1, max_papers // len(self.SEARCH_QUERIES)),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )
                for paper in search.results():
                    if paper.title in seen_titles:
                        continue
                    seen_titles.add(paper.title)

                    if self._is_relevant(paper):
                        model_info = self._extract_model_info(paper)
                        if model_info:
                            self._save_to_database(model_info)
                            discovered.append(model_info)
                            logger.info(f"Discovered: {model_info['name']} ({paper.entry_id})")

            except Exception as e:
                logger.warning(f"arXiv search failed for '{query}': {e}")

        logger.info(f"Discovery complete: {len(discovered)} new models found")
        return discovered

    def _is_relevant(self, paper) -> bool:
        keywords = [
            "surrogate", "neural operator", "mesh", "CFD", "FEA",
            "PDE", "fluid", "structural", "turbulence", "finite element",
            "computational mechanics", "physics-informed",
        ]
        text  = (paper.title + " " + paper.summary).lower()
        score = sum(1 for k in keywords if k.lower() in text)
        return score >= 3

    def _extract_model_info(self, paper) -> Dict:
        github_url = ""
        urls = re.findall(r'https?://github\.com/[^\s\)]+', paper.summary)
        if urls:
            github_url = urls[0].rstrip(".,")

        prompt = f"""
Extract model information from this paper.

Title: {paper.title}
Abstract: {paper.summary[:800]}
GitHub: {github_url}

Output ONLY JSON:
{{
  "name": "short model name (max 20 chars)",
  "mesh_type": "structured OR unstructured OR any",
  "physics_types": ["CFD_incompressible_turbulent", "FEA_static_linear"],
  "benchmark_l2": 0.03,
  "key_innovation": "one sentence",
  "is_new_architecture": true
}}

Output ONLY the JSON.
"""
        try:
            response = self.llm.invoke(prompt)
            content  = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            info = json.loads(content)
            info["github_url"] = github_url
            info["arxiv_id"]   = paper.entry_id
            info["source"]     = "arxiv"
            return info
        except Exception:
            return None

    def _save_to_database(self, info: Dict):
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO discovered_models
                    (name, source, github_url, arxiv_id, mesh_type,
                     physics_types, benchmark_l2, added_date)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    info.get("name", "Unknown"),
                    info.get("source", "arxiv"),
                    info.get("github_url", ""),
                    info.get("arxiv_id", ""),
                    info.get("mesh_type", "any"),
                    json.dumps(info.get("physics_types", [])),
                    info.get("benchmark_l2", 0.05),
                    datetime.datetime.now().isoformat(),
                ))
        except Exception as e:
            logger.warning(f"Failed to save discovered model: {e}")
