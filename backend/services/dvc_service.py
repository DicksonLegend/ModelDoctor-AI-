"""
DVC Service — dataset versioning helpers.
Wraps DVC CLI commands for tracking dataset changes.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from app.config import DATA_RAW_DIR, BASE_DIR


def is_dvc_installed() -> bool:
    """Check if DVC is available on the system."""
    return shutil.which("dvc") is not None


def init_dvc():
    """Initialize DVC in the backend directory if not already initialized."""
    dvc_dir = BASE_DIR / ".dvc"
    if dvc_dir.exists():
        return True

    try:
        subprocess.run(
            ["dvc", "init"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def version_dataset(file_path: Path) -> dict:
    """
    Add a dataset file to DVC tracking.
    Returns status dict.
    """
    if not is_dvc_installed():
        return {"status": "skipped", "reason": "DVC not installed"}

    try:
        result = subprocess.run(
            ["dvc", "add", str(file_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        }
    except Exception as e:
        return {"status": "failed", "reason": str(e)}


def get_dvc_status() -> dict:
    """Get current DVC status."""
    if not is_dvc_installed():
        return {"status": "not_installed"}

    try:
        result = subprocess.run(
            ["dvc", "status"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "status": "ok",
            "output": result.stdout,
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}
