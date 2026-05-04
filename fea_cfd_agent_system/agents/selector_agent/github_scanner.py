"""
GitHub Scanner — reads repository code to understand model capabilities.
Detects mesh type assumptions, dependency conflicts, known issues.
"""

import os
import base64
import requests
from loguru import logger


class GitHubScanner:
    """
    Scans GitHub repos to extract:
    - Stars and last commit (code maturity)
    - Mesh type assumption (reads model.py for grid_sample, edge_index)
    - Input/output shapes
    - Known issues
    - Dependencies
    """

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN", "")
        self.headers = {}
        if self.token:
            self.headers = {"Authorization": f"token {self.token}"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def scan(self, github_url: str) -> dict:
        """Scan a GitHub repository and return a report dict."""
        try:
            owner_repo = github_url.replace("https://github.com/", "").strip("/")
            api_base = f"https://api.github.com/repos/{owner_repo}"

            meta = self.session.get(api_base, timeout=10).json()
            report = {
                "stars":           meta.get("stargazers_count", 0),
                "last_commit":     meta.get("pushed_at", ""),
                "open_issues":     meta.get("open_issues_count", 0),
                "language":        meta.get("language", "Python"),
                "mesh_assumption": "UNKNOWN",
                "dependencies":    [],
                "known_issues":    [],
            }

            # Read README
            try:
                readme_resp = self.session.get(f"{api_base}/readme", timeout=10).json()
                if "content" in readme_resp:
                    readme_text = base64.b64decode(readme_resp["content"]).decode("utf-8", errors="ignore")
                    report["readme_summary"] = readme_text[:2000]
            except Exception:
                pass

            # Read requirements
            try:
                req_url = f"https://raw.githubusercontent.com/{owner_repo}/main/requirements.txt"
                req = self.session.get(req_url, timeout=10)
                if req.status_code == 200:
                    report["dependencies"] = req.text.splitlines()[:20]
            except Exception:
                pass

            # Find and read model file to detect mesh assumptions
            try:
                tree = self.session.get(
                    f"{api_base}/git/trees/main?recursive=1", timeout=10
                ).json()
                model_files = [
                    f["path"] for f in tree.get("tree", [])
                    if "model" in f["path"].lower() and f["path"].endswith(".py")
                ]
                if model_files:
                    code_url = f"https://raw.githubusercontent.com/{owner_repo}/main/{model_files[0]}"
                    code_resp = self.session.get(code_url, timeout=10)
                    if code_resp.status_code == 200:
                        code = code_resp.text
                        report["mesh_assumption"] = self._detect_mesh_type(code)
            except Exception:
                pass

            # Get open issues titles
            try:
                issues = self.session.get(
                    f"{api_base}/issues?state=open&per_page=10", timeout=10
                ).json()
                if isinstance(issues, list):
                    report["known_issues"] = [i["title"] for i in issues[:5] if isinstance(i, dict)]
            except Exception:
                pass

            return report

        except Exception as e:
            logger.warning(f"GitHub scan error for {github_url}: {e}")
            return {"stars": 0, "last_commit": "", "mesh_assumption": "UNKNOWN"}

    def _detect_mesh_type(self, code: str) -> str:
        """Analyze model code to detect mesh type requirements."""
        if "grid_sample" in code or "F.grid_sample" in code:
            return "STRUCTURED_ONLY"
        if "torch.fft" in code and "reshape" in code and "edge_index" not in code:
            return "STRUCTURED_ONLY"
        if "edge_index" in code or "torch_geometric" in code:
            return "UNSTRUCTURED_GNN"
        if "knn" in code.lower() or "ball_query" in code:
            return "POINT_CLOUD"
        if "n_modes" in code and "FNO" in code:
            return "STRUCTURED_ONLY"
        return "UNKNOWN"
