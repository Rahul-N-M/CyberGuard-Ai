from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_project_root(project_root: str | Path | None = None) -> Path:
    """Return the repo root and change into it if available."""
    if project_root is not None:
        root = Path(project_root).resolve()
        os.chdir(root)
        return root

    candidates = [
        Path.cwd(),
        Path("/content/cyberguard-ai"),
        Path("/workspace"),
        Path("/content"),
    ]

    for candidate in candidates:
        if (candidate / "data" / "processed" / "risk_features.csv").exists():
            os.chdir(candidate)
            return candidate.resolve()

    return Path.cwd().resolve()


def install_missing_dependencies(packages: list[str] | None = None) -> None:
    """Install only required ML dependencies when they are absent in Colab."""
    packages = packages or [
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "plotly",
        "google-or-tools",
    ]

    missing = [pkg for pkg in packages if not is_module_installed(pkg)]
    if not missing:
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def is_module_installed(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def git_clone_repository(repo_url: str | None = None) -> Path:
    """Clone the GitHub repository in Colab when needed."""
    if repo_url is None:
        repo_url = os.getenv("GITHUB_REPO_URL")

    if not repo_url:
        return ensure_project_root()

    repo_dir = Path("/content") / "cyberguard-ai"
    if not repo_dir.exists():
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
    os.chdir(repo_dir)
    return repo_dir.resolve()


def project_setup(repo_url: str | None = None) -> Path:
    """Reusable Colab setup flow for notebooks and scripts."""
    root = git_clone_repository(repo_url) if "google.colab" in sys.modules else ensure_project_root()
    install_missing_dependencies()
    return root
