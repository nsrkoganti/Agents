"""
Dataset Search Agent — discovers publicly available CFD/FEA datasets
from HuggingFace Hub, GitHub, and Zenodo that match the current problem.
"""

import json
import datetime
from typing import List, Dict, Optional
from loguru import logger
from agents.shared.llm_factory import get_dev_llm

# Curated high-quality CFD/FEA datasets (always checked first)
CURATED_DATASETS = [
    {
        "source":       "huggingface",
        "repo_id":      "chen-yingfa/CFDBench",
        "name":         "CFDBench",
        "physics_types": ["cfd_incompressible", "CFD_incompressible"],
        "mesh_type":    "structured",
        "n_samples":    10000,
        "size_gb":      14.4,
        "license":      "Apache-2.0",
        "format":       "parquet",
        "description":  "Large-scale incompressible CFD benchmark dataset",
    },
    {
        "source":       "huggingface",
        "repo_id":      "nvidia/PhysicsNeMo-Datacenter-CFD",
        "name":         "PhysicsNeMo-Datacenter-CFD",
        "physics_types": ["cfd_incompressible", "CFD_incompressible_turbulent"],
        "mesh_type":    "unstructured",
        "n_samples":    5000,
        "size_gb":      8.0,
        "license":      "Apache-2.0",
        "format":       "vtk",
        "description":  "OpenFOAM datacenter cooling CFD simulations",
    },
    {
        "source":       "huggingface",
        "repo_id":      "nvidia/PhysicsNeMo-CFD-Ahmed-Body",
        "name":         "PhysicsNeMo-CFD-Ahmed-Body",
        "physics_types": ["cfd_incompressible", "CFD_incompressible_turbulent"],
        "mesh_type":    "unstructured",
        "n_samples":    500,
        "size_gb":      12.0,
        "license":      "Apache-2.0",
        "format":       "vtk",
        "description":  "Ahmed body 3D aerodynamics dataset",
    },
    {
        "source":       "huggingface",
        "repo_id":      "Extrality/AirfRANS",
        "name":         "AirfRANS",
        "physics_types": ["cfd_incompressible", "CFD_incompressible"],
        "mesh_type":    "unstructured",
        "n_samples":    1000,
        "size_gb":      2.0,
        "license":      "Apache-2.0",
        "format":       "vtu",
        "description":  "NACA airfoil RANS surrogate dataset, VTU format",
    },
    {
        "source":       "huggingface",
        "repo_id":      "CFDML/TransportBench",
        "name":         "TransportBench",
        "physics_types": ["cfd_incompressible", "CFD_incompressible", "thermal"],
        "mesh_type":    "unstructured",
        "n_samples":    8000,
        "size_gb":      5.0,
        "license":      "MIT",
        "format":       "hdf5",
        "description":  "Transport equation benchmark dataset",
    },
    {
        "source":       "github",
        "repo_url":     "https://github.com/thunil/Deep-Flow-Prediction",
        "name":         "Deep-Flow-Prediction",
        "physics_types": ["cfd_incompressible", "CFD_incompressible"],
        "mesh_type":    "structured",
        "n_samples":    2000,
        "size_gb":      1.5,
        "license":      "MIT",
        "format":       "numpy",
        "description":  "2D RANS flow field prediction dataset",
    },
]


class DatasetSearchAgent:
    """
    Searches HuggingFace, GitHub, and Zenodo for datasets matching
    the current problem card (physics type, mesh type, scale).
    """

    HF_SEARCH_URL   = "https://huggingface.co/api/datasets"
    ZENODO_URL       = "https://zenodo.org/api/records"
    GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

    def __init__(self, config: dict):
        self.config = config
        self.llm    = get_dev_llm(max_tokens=1500)

    def search(self, physics_type: str, mesh_type: str,
               min_samples: int = 100,
               problem_description: str = "") -> List[Dict]:
        """
        Search all sources and return ranked list of datasets.
        Always starts with curated datasets, augments with live search.
        """
        logger.info(f"Dataset search: physics={physics_type}, mesh={mesh_type}")

        # 1. Filter curated datasets by physics type
        candidates = self._filter_curated(physics_type, mesh_type)
        logger.info(f"  {len(candidates)} curated dataset(s) match physics type")

        # 2. Search HuggingFace
        hf_results = self._search_huggingface(physics_type)
        candidates.extend(hf_results)
        logger.info(f"  {len(hf_results)} result(s) from HuggingFace")

        # 3. Search Zenodo
        zenodo_results = self._search_zenodo(physics_type)
        candidates.extend(zenodo_results)
        logger.info(f"  {len(zenodo_results)} result(s) from Zenodo")

        # 4. Deduplicate by name
        seen  = set()
        unique = []
        for d in candidates:
            key = d.get("repo_id") or d.get("repo_url") or d.get("name")
            if key and key not in seen:
                seen.add(key)
                unique.append(d)

        # 5. LLM ranking
        ranked = self._rank_with_llm(unique, physics_type, mesh_type,
                                     min_samples, problem_description)
        logger.info(f"Dataset search complete: {len(ranked)} candidate(s) ranked")
        return ranked

    def _filter_curated(self, physics_type: str, mesh_type: str) -> List[Dict]:
        results = []
        for d in CURATED_DATASETS:
            types = d.get("physics_types", [])
            mesh  = d.get("mesh_type", "any")
            if any(physics_type in t or t in physics_type for t in types):
                if mesh == "any" or mesh_type in mesh or mesh in mesh_type:
                    results.append(dict(d))
        return results or list(CURATED_DATASETS[:3])  # fallback: top 3

    def _search_huggingface(self, physics_type: str) -> List[Dict]:
        try:
            import requests
            query = self._physics_to_query(physics_type)
            resp  = requests.get(
                self.HF_SEARCH_URL,
                params={"search": query, "sort": "downloads", "limit": 10},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            results = []
            for item in resp.json():
                if not isinstance(item, dict):
                    continue
                repo_id = item.get("id", "")
                if not repo_id:
                    continue
                results.append({
                    "source":       "huggingface",
                    "repo_id":      repo_id,
                    "name":         repo_id.split("/")[-1],
                    "description":  item.get("description", ""),
                    "n_downloads":  item.get("downloads", 0),
                    "license":      item.get("cardData", {}).get("license", "unknown"),
                    "physics_types": [physics_type],
                    "mesh_type":    "unknown",
                    "format":       "unknown",
                    "n_samples":    0,
                    "size_gb":      0,
                })
            return results
        except Exception as e:
            logger.debug(f"HuggingFace search failed: {e}")
            return []

    def _search_zenodo(self, physics_type: str) -> List[Dict]:
        try:
            import requests
            query = self._physics_to_query(physics_type)
            resp  = requests.get(
                self.ZENODO_URL,
                params={"q": query, "type": "dataset", "size": 5},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            results = []
            for item in resp.json().get("hits", {}).get("hits", []):
                meta = item.get("metadata", {})
                files = item.get("files", [])
                results.append({
                    "source":       "zenodo",
                    "zenodo_id":    str(item.get("id", "")),
                    "name":         meta.get("title", "unknown")[:60],
                    "description":  meta.get("description", "")[:200],
                    "license":      meta.get("license", {}).get("id", "unknown"),
                    "physics_types": [physics_type],
                    "mesh_type":    "unknown",
                    "format":       files[0].get("type", "unknown") if files else "unknown",
                    "n_samples":    0,
                    "size_gb":      sum(f.get("size", 0) for f in files) / 1e9,
                    "download_url": files[0].get("links", {}).get("self") if files else None,
                })
            return results
        except Exception as e:
            logger.debug(f"Zenodo search failed: {e}")
            return []

    def _rank_with_llm(self, datasets: List[Dict], physics_type: str,
                        mesh_type: str, min_samples: int,
                        problem_description: str) -> List[Dict]:
        if not datasets:
            return []
        if len(datasets) == 1:
            return datasets

        summary = []
        for i, d in enumerate(datasets[:15]):
            summary.append({
                "index":   i,
                "name":    d.get("name"),
                "source":  d.get("source"),
                "license": d.get("license", "?"),
                "samples": d.get("n_samples", 0),
                "size_gb": d.get("size_gb", 0),
                "format":  d.get("format", "?"),
                "desc":    d.get("description", "")[:100],
            })

        prompt = f"""
You are a CFD/FEA data scientist selecting the best training dataset.

Problem: {physics_type} on {mesh_type} mesh.
{f'Context: {problem_description}' if problem_description else ''}
Minimum samples needed: {min_samples}

Available datasets:
{json.dumps(summary, indent=2)}

Rank them by suitability. Prefer: open license (MIT/Apache/CC-BY), large n_samples,
standard formats (VTK/HDF5/NPZ), known-good sources (HuggingFace official, nvidia, CFDML).
Exclude CC-BY-NC or unknown licenses.

Output ONLY JSON array of indices in ranked order (best first), max 5:
[2, 0, 4, ...]
"""
        try:
            resp    = self.llm.invoke(prompt)
            content = resp.content.strip()
            if "[" in content:
                content = content[content.index("["):content.rindex("]") + 1]
            indices = json.loads(content)
            ranked  = []
            for i in indices:
                if 0 <= i < len(datasets):
                    ranked.append(datasets[i])
            # Append anything not in ranked list
            ranked_set = {id(d) for d in ranked}
            for d in datasets:
                if id(d) not in ranked_set:
                    ranked.append(d)
            return ranked
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e} — using original order")
            return datasets

    def _physics_to_query(self, physics_type: str) -> str:
        mapping = {
            "cfd":          "CFD fluid dynamics surrogate neural network",
            "incompressible": "incompressible CFD flow surrogate",
            "turbulent":    "turbulent flow RANS surrogate machine learning",
            "fea":          "finite element FEA structural surrogate",
            "thermal":      "thermal heat transfer CFD surrogate",
            "compressible": "compressible flow CFD surrogate",
        }
        pt = physics_type.lower()
        for key, query in mapping.items():
            if key in pt:
                return query
        return f"CFD FEA simulation surrogate {physics_type}"
