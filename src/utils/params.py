"""
Parameter accounting + matched-capacity / frozen-encoder invariants.

- total      : the matched-capacity claim (identical across the four systems).
- active     : params of the branches whose modality is active for this system,
               plus the fusion head -- the honest "what actually does work" number
               (disabled branches are present but fed zeros).
- frozen     : must be 0 -- encoders (WavLM/RoBERTa) are offline-cached, never in
               the training graph.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

# modality name (in GroupTurnFuse.active) -> branch attribute / named_child
BRANCH_OF_MODALITY = {
    "timing": "frame_branch",
    "scalar": "scalar_branch",
    "audio": "audio_branch",
    "text": "text_branch",
}


def count_parameters(model: nn.Module) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    per_branch = {name: sum(p.numel() for p in module.parameters())
                  for name, module in model.named_children()}
    return {"total": total, "trainable": trainable, "frozen": frozen,
            "per_branch": per_branch}


def active_parameters(model: nn.Module) -> int:
    """Params of active-modality branches + the fusion head."""
    per_branch = {name: sum(p.numel() for p in module.parameters())
                  for name, module in model.named_children()}
    active_set = getattr(model, "active", set(BRANCH_OF_MODALITY))
    branches = {BRANCH_OF_MODALITY[m] for m in active_set if m in BRANCH_OF_MODALITY}
    branches.add("fusion")
    return int(sum(per_branch.get(b, 0) for b in branches))


def params_summary(model: nn.Module) -> Dict:
    s = count_parameters(model)
    s["active"] = active_parameters(model)
    return s


def assert_frozen_zero(model: nn.Module) -> None:
    """Encoders must be offline-cached, not in the graph -> no frozen params."""
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if frozen != 0:
        names = [n for n, p in model.named_parameters() if not p.requires_grad]
        raise AssertionError(
            f"{frozen} frozen param(s) in the model graph (expected 0; encoders must be "
            f"offline-cached, never instantiated in training). Offending: {names[:8]}"
        )


def format_params_line(system: str, model: nn.Module) -> str:
    s = params_summary(model)
    return (f"params | system={system} | total={s['total']:,} | active={s['active']:,} | "
            f"trainable={s['trainable']:,} | frozen={s['frozen']:,}")
