"""Atomic checkpoint save/load (tmp -> rename), with full RNG + optimizer state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(state: Dict, path) -> None:
    """Write atomically: torch.save to a tmp file then os.replace (rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path, map_location="cpu") -> Dict:
    # weights_only=False: our checkpoints contain RNG states / config (not just tensors)
    return torch.load(path, map_location=map_location, weights_only=False)
