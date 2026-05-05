"""
Dataset Download Agent — downloads and caches datasets from HuggingFace,
GitHub, and Zenodo. Skips download if cache already exists.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


CACHE_DIR = Path("data/downloads")


class DatasetDownloadAgent:
    """
    Downloads datasets to a local cache directory.
    Supports: HuggingFace snapshot, GitHub clone, Zenodo/URL HTTP download.
    """

    def __init__(self, config: dict):
        self.config    = config
        self.cache_dir = Path(config.get("dataset_cache_dir", str(CACHE_DIR)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, dataset_info: Dict) -> Optional[str]:
        """
        Download dataset described by dataset_info dict.
        Returns local path to downloaded data, or None on failure.
        """
        source = dataset_info.get("source", "")
        name   = dataset_info.get("name", "unknown").replace("/", "__")
        local  = self.cache_dir / name

        if local.exists() and any(local.iterdir()):
            logger.info(f"Dataset cache hit: {local}")
            return str(local)

        logger.info(f"Downloading dataset '{name}' from {source}...")

        try:
            if source == "huggingface":
                return self._download_huggingface(dataset_info, local)
            elif source == "github":
                return self._download_github(dataset_info, local)
            elif source in ("zenodo", "figshare", "url"):
                return self._download_url(dataset_info, local)
            else:
                logger.warning(f"Unknown source '{source}' — attempting URL download")
                return self._download_url(dataset_info, local)
        except Exception as e:
            logger.error(f"Download failed for '{name}': {e}")
            return None

    def _download_huggingface(self, info: Dict, local: Path) -> Optional[str]:
        repo_id = info.get("repo_id")
        if not repo_id:
            return None
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local),
                ignore_patterns=["*.bin", "*.safetensors"],  # skip model weights
            )
            logger.success(f"HuggingFace download complete: {path}")
            return path
        except ImportError:
            logger.warning("huggingface_hub not installed — pip install huggingface_hub")
            return self._download_hf_fallback(info, local)
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return None

    def _download_hf_fallback(self, info: Dict, local: Path) -> Optional[str]:
        """Fallback: use datasets library."""
        try:
            from datasets import load_dataset
            repo_id = info.get("repo_id")
            ds = load_dataset(repo_id, split="train", streaming=False)
            local.mkdir(parents=True, exist_ok=True)
            out = str(local / "data.parquet")
            ds.to_parquet(out)
            logger.success(f"datasets library download complete: {out}")
            return str(local)
        except Exception as e:
            logger.error(f"datasets fallback failed: {e}")
            return None

    def _download_github(self, info: Dict, local: Path) -> Optional[str]:
        repo_url = info.get("repo_url")
        if not repo_url:
            return None
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(local)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.success(f"Git clone complete: {local}")
                return str(local)
            logger.error(f"Git clone failed: {result.stderr}")
            return None
        except Exception as e:
            logger.error(f"GitHub download failed: {e}")
            return None

    def _download_url(self, info: Dict, local: Path) -> Optional[str]:
        url = info.get("download_url") or info.get("url")
        if not url:
            return None
        try:
            import requests
            local.mkdir(parents=True, exist_ok=True)
            filename = url.split("/")[-1] or "data"
            dest = local / filename

            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (50 * 1024 * 1024) == 0:
                            pct = 100 * downloaded / total
                            logger.info(f"  Download progress: {pct:.0f}%")

            logger.success(f"URL download complete: {dest}")
            return str(local)
        except Exception as e:
            logger.error(f"URL download failed: {e}")
            return None

    def get_cache_path(self, dataset_info: Dict) -> Optional[str]:
        """Check if dataset is already cached without downloading."""
        name  = dataset_info.get("name", "").replace("/", "__")
        local = self.cache_dir / name
        if local.exists() and any(local.iterdir()):
            return str(local)
        return None
