"""Centralized Configuration Loader and Manager for VulnShield-DNN.

Provides robust YAML loading, dot-notation access, variable resolution, and schema validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from vulnshield.core.exceptions import ConfigurationError


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style access (dot notation)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get nested value using dot-delimited path (e.g., 'training.optimizer.lr')."""
        keys = path.split(".")
        curr: Any = self
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr


def load_yaml(file_path: Union[str, Path]) -> ConfigDict:
    """Load a YAML configuration file into a ConfigDict.

    Args:
        file_path: Path to the YAML file.

    Returns:
        ConfigDict containing configuration data.

    Raises:
        ConfigurationError: If file does not exist or YAML parsing fails.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML file root must be a mapping/dict, got {type(data).__name__} in {path}")

    return _to_config_dict(data)


def save_yaml(data: Union[Dict[str, Any], ConfigDict], file_path: Union[str, Path]) -> None:
    """Save configuration dictionary to a YAML file.

    Args:
        data: Dictionary or ConfigDict to save.
        file_path: Target path for the YAML file.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    plain_dict = _to_plain_dict(data)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(plain_dict, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ConfigurationError(f"Failed to write YAML file {path}: {e}") from e


def merge_configs(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge override ConfigDict into base ConfigDict."""
    result = ConfigDict(_to_plain_dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(ConfigDict(result[key]), ConfigDict(value))
        else:
            result[key] = value
    return _to_config_dict(result)


def resolve_project_paths(config: ConfigDict, project_root: Optional[Union[str, Path]] = None) -> ConfigDict:
    """Resolve '{project_root}' string templates in configuration values."""
    if project_root is None:
        # Default project root is 3 levels above this file: src/vulnshield/utils/config.py -> repo_root
        root_path = Path(__file__).resolve().parent.parent.parent.parent
        root_str = str(root_path).replace("\\", "/")
    elif isinstance(project_root, Path):
        root_str = str(project_root.resolve()).replace("\\", "/")
    else:
        root_str = str(project_root).replace("\\", "/")

    def _resolve(val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("{project_root}", root_str)
        elif isinstance(val, dict):
            return {k: _resolve(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_resolve(item) for item in val]
        return val

    plain = _to_plain_dict(config)
    resolved = _resolve(plain)
    return _to_config_dict(resolved)


def _to_config_dict(d: Dict[str, Any]) -> ConfigDict:
    """Recursively convert nested dicts to ConfigDict."""
    res = ConfigDict()
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _to_config_dict(v)
        elif isinstance(v, list):
            res[k] = [
                _to_config_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            res[k] = v
    return res


def _to_plain_dict(d: Union[Dict[str, Any], ConfigDict]) -> Dict[str, Any]:
    """Recursively convert ConfigDict to standard Python dict."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _to_plain_dict(v)
        elif isinstance(v, list):
            res[k] = [
                _to_plain_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            res[k] = v
    return res
