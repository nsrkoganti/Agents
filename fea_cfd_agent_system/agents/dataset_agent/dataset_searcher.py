"""
Dataset Search Agent — discovers publicly available FEA datasets
from HuggingFace Hub, GitHub, and Zenodo that match the current problem.
"""

import json
import datetime
from typing import List, Dict
from loguru import logger
from agents.shared.llm_factory import get_dev_llm


# Curated high-quality FEA datasets (always checked first)
CURATED_DATASETS = [
    {
        "name":         "Transolver-Industrial-FEA",
        "source":       "github",
        "repo_url":     "https://github.com/thuml/Transolver",
        "physics_types": ["FEA_static_linear", "FEA_static_nonlinear"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "MIT",
        "format":       "numpy",
        "description":  "Industrial-scale FEA dataset used for Transolver/Transolver++ benchmarks",
    },
    {
        "name":         "EAGNN-Mesh-Surrogate",
        "source":       "github",
        "repo_url":     "https://github.com/aravi11/EAGNN",
        "physics_types": ["FEA_static_linear"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "MIT",
        "format":       "numpy",
        "description":  "Unstructured mesh stress/displacement dataset for EAGNN benchmarks",
    },
    {
        "name":         "DeepXDE-FEA-Benchmarks",
        "source":       "github",
        "repo_url":     "https://github.com/lululxvi/deepxde",
        "physics_types": ["FEA_static_linear", "FEA_static_nonlinear"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "LGPL-2.1",
        "format":       "numpy",
        "description":  "PINN / DeepONet FEA benchmark problems including Euler beam, plate bending",
    },
    {
        "name":         "DeepFEA-Transient-Dataset",
        "source":       "zenodo",
        "physics_types": ["FEA_dynamic"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "cc-by-4.0",
        "format":       "hdf5",
        "description":  "Transient / dynamic FEA training dataset for ConvLSTM surrogates",
    },
    {
        "name":         "SMT-Structural-Benchmarks",
        "source":       "github",
        "repo_url":     "https://github.com/SMTorg/smt",
        "physics_types": ["FEA_static_linear"],
        "mesh_type":    "structured",
        "license":      "BSD",
        "format":       "numpy",
        "description":  "Structural benchmark problems from SMT toolbox (Kriging, RBF baselines)",
    },
    {
        "name":         "CalculiX-Benchmark-Cases",
        "source":       "github",
        "repo_url":     "https://github.com/calculix/pygccx",
        "physics_types": ["FEA_static_linear", "thermal"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "GPL-2.0",
        "format":       "frd",
        "description":  "Standard CalculiX FEA benchmark cases (.frd format)",
    },
    {
        "name":         "DeepJEB-Jet-Engine-Bracket",
        "source":       "huggingface",
        "hf_dataset_id": "youngvice/DeepJEB",
        "physics_types": ["FEA_static_linear", "FEA_static_nonlinear"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "cc-by-4.0",
        "format":       "stl+csv",
        "n_samples":    2138,
        "description":  "2,138 jet engine bracket designs with structural FEA results (arXiv:2406.09047)",
    },
    {
        "name":         "SimuStruct-Plates",
        "source":       "zenodo",
        "physics_types": ["FEA_static_linear"],
        "mesh_type":    "unstructured_tetrahedral",
        "license":      "cc-by-4.0",
        "format":       "numpy",
        "n_samples":    1000,
        "description":  "1,000 plates with 6 circular holes; MeshGraphNet benchmark dataset",
    },
    {
        "name":         "HEMEW-3D-Elastic-Waves",
        "source":       "zenodo",
        "physics_types": ["FEA_dynamic"],
        "mesh_type":    "structured",
        "license":      "cc-by-4.0",
        "format":       "hdf5",
        "n_samples":    30000,
        "description":  "30,000 3D elastic wavefields; Factorized FNO benchmark (HEMEW-3D database)",
    },
]


class DatasetSearchAgent:
    """
    Searches HuggingFace, GitHub, and Zenodo for FEA datasets matching
    the current problem card (physics type, mesh type, scale).
    """

    HF_SEARCH_URL    = "https://huggingface.co/api/datasets"
    ZENODO_URL       = "https://zenodo.org/api/records"

    def __init__(self, config: dict):
        self.config = config
        self.llm    = get_dev_llm(max_tokens=1500)

    def search(self, physics_type: str, mesh_type: str,
               min_samples: int = 100,
               problem_description: str = "") -> List[Dict]:
        """
        Search all sources and return ranked list of FEA datasets.
        Always starts with curated datasets, augments with live search.
        """
        logger.info(f"FEA Dataset search: physics={physics_type}, mesh={mesh_type}")

        # 1. Filter curated datasets
        candidates = self._filter_curated(physics_type, mesh_type)
        logger.info(f"  {len(candidates)} curated FEA dataset(s) match")

        # 2. HuggingFace search
        hf_results = self._search_huggingface(physics_type)
        candidates.extend(hf_results)
        logger.info(f"  {len(hf_results)} result(s) from HuggingFace")

        # 3. Zenodo search
        zenodo_results = self._search_zenodo(physics_type)
        candidates.extend(zenodo_results)
        logger.info(f"  {len(zenodo_results)} result(s) from Zenodo")

        # Deduplicate by name/repo
        seen, unique = set(), []
        for d in candidates:
            key = d.get("hf_dataset_id") or d.get("repo_url") or d.get("name")
            if key and key not in seen:
                seen.add(key)
                unique.append(d)

        # LLM ranking
        ranked = self._rank_with_llm(unique, physics_type, mesh_type,
                                     min_samples, problem_description)
        logger.info(f"FEA dataset search complete: {len(ranked)} candidate(s)")
        return ranked

    def _filter_curated(self, physics_type: str, mesh_type: str) -> List[Dict]:
        results = []
        for d in CURATED_DATASETS:
            types = d.get("physics_types", [])
            if any(physics_type.lower() in t.lower() or t.lower() in physics_type.lower()
                   for t in types):
                results.append(dict(d))
        return results or list(CURATED_DATASETS[:3])

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
                    "source":        "huggingface",
                    "hf_dataset_id": repo_id,
                    "name":          repo_id.split("/")[-1],
                    "description":   item.get("description", ""),
                    "n_downloads":   item.get("downloads", 0),
                    "license":       item.get("cardData", {}).get("license", "unknown"),
                    "physics_types": [physics_type],
                    "mesh_type":     "unknown",
                    "format":        "unknown",
                    "n_samples":     0,
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
                meta  = item.get("metadata", {})
                files = item.get("files", [])
                results.append({
                    "source":        "zenodo",
                    "zenodo_id":     str(item.get("id", "")),
                    "name":          meta.get("title", "unknown")[:60],
                    "description":   meta.get("description", "")[:200],
                    "license":       meta.get("license", {}).get("id", "unknown"),
                    "physics_types": [physics_type],
                    "mesh_type":     "unknown",
                    "format":        files[0].get("type", "unknown") if files else "unknown",
                    "n_samples":     0,
                    "size_gb":       sum(f.get("size", 0) for f in files) / 1e9,
                    "download_url":  files[0].get("links", {}).get("self") if files else None,
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

        summary = [
            {
                "index":   i,
                "name":    d.get("name"),
                "source":  d.get("source"),
                "license": d.get("license", "?"),
                "samples": d.get("n_samples", 0),
                "format":  d.get("format", "?"),
                "desc":    d.get("description", "")[:100],
            }
            for i, d in enumerate(datasets[:15])
        ]
        prompt = f"""
You are an FEA ML data scientist selecting the best training dataset.

Problem: {physics_type} on {mesh_type} mesh.
{f'Context: {problem_description}' if problem_description else ''}
Minimum samples needed: {min_samples}

Available FEA datasets:
{json.dumps(summary, indent=2)}

Rank them by suitability for training an FEA surrogate model.
Prefer: open license (MIT/Apache/CC-BY), large n_samples, standard formats (VTK/HDF5/NPZ),
known-good sources (thuml/Transolver, DeepXDE, SMTorg), displacement+stress fields present.
Exclude proprietary or CC-BY-NC licenses.

Output ONLY JSON array of indices in ranked order (best first), max 5:
[2, 0, 4, ...]
"""
        try:
            resp    = self.llm.invoke(prompt)
            content = resp.content.strip()
            if "[" in content:
                content = content[content.index("["):content.rindex("]") + 1]
            indices = json.loads(content)
            ranked  = [datasets[i] for i in indices if 0 <= i < len(datasets)]
            seen_id = {id(d) for d in ranked}
            ranked += [d for d in datasets if id(d) not in seen_id]
            return ranked
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e} — using original order")
            return datasets

    def _physics_to_query(self, physics_type: str) -> str:
        mapping = {
            "fea_static_linear":    "finite element structural stress displacement surrogate neural",
            "fea_static_nonlinear": "nonlinear FEA large deformation plasticity surrogate ML",
            "fea_dynamic":          "transient structural dynamics FEA deep learning surrogate",
            "thermal":              "thermal FEA heat transfer conduction surrogate",
            "thermal_structural":   "thermo-mechanical coupled FEA surrogate",
            "multiphysics":         "multiphysics FEA coupled structural thermal surrogate",
        }
        pt = physics_type.lower().replace("-", "_")
        for key, query in mapping.items():
            if key in pt:
                return query
        return f"finite element analysis FEA surrogate neural network {physics_type}"
