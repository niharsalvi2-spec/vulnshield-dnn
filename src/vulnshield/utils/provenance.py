"""Scientific Experiment Provenance, Metadata Tracking and Reproducibility Manifest System."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute cryptographic SHA-256 checksum of a file for immutable artifact verification."""
    p = Path(filepath).resolve()
    if not p.exists() or not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_revision_info(repo_dir: Union[str, Path]) -> Dict[str, Union[str, bool]]:
    """Retrieve current Git commit SHA and dirty working tree status."""
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()

        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
        return {"commit_sha": commit_sha, "is_dirty": is_dirty}
    except Exception:
        return {"commit_sha": "unavailable", "is_dirty": True}


@dataclass
class ExperimentManifest:
    """Immutable scientific metadata manifest accompanying every experimental output."""
    experiment_id: str
    stage_name: str
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    git_info: Dict[str, Union[str, bool]] = field(default_factory=dict)
    system_env: Dict[str, Any] = field(default_factory=dict)
    reproducibility_seed: int = 42
    parameters: Dict[str, Any] = field(default_factory=dict)
    cli_arguments: Dict[str, Any] = field(default_factory=dict)
    output_artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> sha256

    @classmethod
    def create(
        cls,
        experiment_id: str,
        stage_name: str,
        seed: int = 42,
        parameters: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None,
        repo_root: Optional[Union[str, Path]] = None
    ) -> "ExperimentManifest":
        """Factory method to capture runtime environment state."""
        root = Path(repo_root).resolve() if repo_root else Path.cwd()
        git_info = get_git_revision_info(root)

        system_env = {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        return cls(
            experiment_id=experiment_id,
            stage_name=stage_name,
            git_info=git_info,
            system_env=system_env,
            reproducibility_seed=seed,
            parameters=parameters or {},
            cli_arguments=cli_args or {}
        )

    def record_artifact(self, filepath: Union[str, Path]) -> None:
        """Register an output artifact with its cryptographic SHA-256 hash."""
        p = Path(filepath).resolve()
        if p.exists() and p.is_file():
            self.output_artifacts[p.name] = compute_file_sha256(p)

    def save(self, destination_dir: Union[str, Path]) -> Path:
        """Serialize the manifest to manifest.json in the designated directory."""
        dest = Path(destination_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        manifest_path = dest / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return manifest_path
