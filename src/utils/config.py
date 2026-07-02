"""YAML config loading with deep-merge (base <- system <- CLI overrides)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(base_path, system_path=None, overrides: Dict = None) -> Dict:
    """Load base config, deep-merge an optional system config and CLI overrides."""
    cfg = load_yaml(base_path)
    if system_path is not None:
        cfg = _deep_merge(cfg, load_yaml(system_path))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def save_yaml(cfg: Dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
