"""Reproducibility: global seeding, deterministic cudnn, seeded DataLoader workers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    """Seed all RNGs and (optionally) request deterministic algorithms.

    strict=False uses warn_only=True so a kernel without a deterministic impl warns
    rather than raising; set strict=True to hard-fail instead. cudnn flags + the
    CUBLAS workspace var make CUDA matmuls reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception:  # pragma: no cover - very old torch
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive each worker's seed from torch's base seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_states(states: dict) -> None:
    if not states:
        return
    if states.get("python") is not None:
        random.setstate(states["python"])
    if states.get("numpy") is not None:
        np.random.set_state(states["numpy"])
    if states.get("torch") is not None:
        t = states["torch"]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.uint8)
        torch.set_rng_state(t.to(torch.uint8) if t.dtype != torch.uint8 else t)
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
