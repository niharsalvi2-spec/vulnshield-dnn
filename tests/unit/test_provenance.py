"""Unit Tests for Experiment Provenance, Manifest and Checksums."""

from pathlib import Path
import json
import pytest

from vulnshield.utils.provenance import (
    compute_file_sha256,
    get_git_revision_info,
    ExperimentManifest
)


@pytest.mark.unit
class TestProvenanceSystem:

    def test_compute_file_sha256(self, tmp_path):
        test_file = tmp_path / "test_artifact.json"
        test_file.write_text('{"accuracy": 93.2}', encoding="utf-8")

        sha = compute_file_sha256(test_file)
        assert len(sha) == 64  # SHA-256 is 64 hex chars
        assert compute_file_sha256(tmp_path / "nonexistent.txt") == ""

    def test_manifest_creation_and_serialization(self, tmp_path):
        manifest = ExperimentManifest.create(
            experiment_id="exp_test_001",
            stage_name="discovery",
            seed=42,
            parameters={"budget": 50},
            cli_args={"model": "resnet18"}
        )

        test_artifact = tmp_path / "discovery_results.json"
        test_artifact.write_text('{"top_channels": []}', encoding="utf-8")
        manifest.record_artifact(test_artifact)

        saved_path = manifest.save(tmp_path)
        assert saved_path.exists()

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["experiment_id"] == "exp_test_001"
        assert data["stage_name"] == "discovery"
        assert "discovery_results.json" in data["output_artifacts"]
        assert len(data["output_artifacts"]["discovery_results.json"]) == 64
