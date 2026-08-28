"""Project Structure Verification Script for VulnShield-DNN.

Validates that all directories, package init files, and configuration directories
match the frozen Repository Architecture v1.0 specifications.
"""

import os
import sys
from pathlib import Path

REQUIRED_DIRECTORIES = [
    # Configs
    "configs/project", "configs/data", "configs/models", "configs/faults",
    "configs/discovery", "configs/baselines", "configs/interaction",
    "configs/adversarial", "configs/protection", "configs/experiments",
    
    # Source packages
    "src/vulnshield/core", "src/vulnshield/data", "src/vulnshield/models/resnet",
    "src/vulnshield/models/vgg", "src/vulnshield/training", "src/vulnshield/fault_injection",
    "src/vulnshield/vulnerability", "src/vulnshield/discovery/td3", "src/vulnshield/discovery/environment",
    "src/vulnshield/baselines/random", "src/vulnshield/baselines/activation",
    "src/vulnshield/baselines/gradient", "src/vulnshield/baselines/taylor",
    "src/vulnshield/baselines/ddpg", "src/vulnshield/interaction", "src/vulnshield/adversarial",
    "src/vulnshield/protection", "src/vulnshield/evaluation", "src/vulnshield/experiments",
    "src/vulnshield/analysis", "src/vulnshield/utils",
    
    # Scripts
    "scripts/setup", "scripts/data", "scripts/models", "scripts/discovery",
    "scripts/interaction", "scripts/adversarial", "scripts/protection",
    "scripts/evaluation", "scripts/pipeline",
    
    # Data & Checkpoints
    "data/raw", "data/interim", "data/processed", "data/metadata/channel_catalogs",
    "checkpoints/base_models/resnet18", "checkpoints/base_models/vgg16",
    "checkpoints/td3/resnet18", "checkpoints/td3/vgg16",
    "checkpoints/protected/resnet18/budget_1pct", "checkpoints/protected/resnet18/budget_3pct",
    "checkpoints/protected/resnet18/budget_5pct", "checkpoints/protected/resnet18/budget_10pct",
    "checkpoints/protected/vgg16/budget_1pct", "checkpoints/protected/vgg16/budget_3pct",
    "checkpoints/protected/vgg16/budget_5pct", "checkpoints/protected/vgg16/budget_10pct",
    
    # Results & Reports
    "results/discovery", "results/vulnerability", "results/interaction",
    "results/adversarial", "results/protection", "results/evaluation", "results/final",
    "reports/exploratory", "reports/experiments", "reports/final",
    
    # Tests, Docs, Artifacts, Logs
    "tests/unit", "tests/integration", "tests/regression", "tests/fixtures",
    "docs/architecture", "docs/methodology", "docs/experiments", "docs/development",
    "artifacts/figures", "artifacts/tables", "artifacts/diagrams",
    "logs/training", "logs/experiments", "logs/system", "temp"
]

REQUIRED_ROOT_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    "pytest.ini",
    "CITATION.cff",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md"
]

def verify_structure(root_path: Path) -> bool:
    print(f"[*] Verifying VulnShield-DNN Repository at: {root_path}")
    missing_dirs = []
    missing_files = []
    
    for rel_dir in REQUIRED_DIRECTORIES:
        p = root_path / rel_dir
        if not p.is_dir():
            missing_dirs.append(rel_dir)
            
    for rel_file in REQUIRED_ROOT_FILES:
        p = root_path / rel_file
        if not p.is_file():
            missing_files.append(rel_file)
            
    if missing_dirs:
        print(f"[FAIL] Missing {len(missing_dirs)} required directories:")
        for d in missing_dirs[:10]:
            print(f"  - {d}")
        if len(missing_dirs) > 10:
            print(f"  ... and {len(missing_dirs) - 10} more.")
            
    if missing_files:
        print(f"[FAIL] Missing {len(missing_files)} required root files:")
        for f in missing_files:
            print(f"  - {f}")
            
    if not missing_dirs and not missing_files:
        print("[PASS] All required directories and governance files exist and conform to Frozen Architecture v1.0.")
        return True
    return False

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    success = verify_structure(repo_root)
    sys.exit(0 if success else 1)
