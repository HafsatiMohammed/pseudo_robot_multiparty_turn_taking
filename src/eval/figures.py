"""
Phase 7 figures: DET curves (all systems incl. rules), confusion matrices,
gate-inspection (mean modality gate within each stratum).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .metrics import LABELS, det_eer_from_scores

_COLORS = ["#1D9E75", "#D85A30", "#3B6EA5", "#9A60B4", "#888780", "#C1A03F", "#444444"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def det_all_systems(trained_scores: Dict[str, Dict], rule_curves: Dict[str, Dict], path) -> None:
    """
    trained_scores[name] = {"y_true": arr, "scores": arr (=1-P(WAIT))}.
    rule_curves[name] = {"false_alarm_rate": arr, "miss_rate": arr, "eer": float}.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ci = 0
    for name, d in trained_scores.items():
        try:
            det = det_eer_from_scores(d["y_true"], d["scores"])
        except ValueError:
            continue
        ax.plot(det["false_alarm_rate"], det["miss_rate"], color=_COLORS[ci % len(_COLORS)],
                label=f"{name} (EER={det['eer']:.2f})")
        ax.plot(det["eer"], det["eer"], "o", color=_COLORS[ci % len(_COLORS)], ms=5)
        ci += 1
    for name, r in rule_curves.items():
        ax.plot(r["false_alarm_rate"], r["miss_rate"], "--", color=_COLORS[ci % len(_COLORS)],
                label=f"{name} (EER={r['eer']:.2f})")
        ci += 1
    ax.plot([0, 1], [0, 1], ":", color="#aaa", lw=1)
    ax.set_xlabel("false-alarm rate (of true WAITs)")
    ax.set_ylabel("miss rate (of true entries)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("DET: WAIT vs entry")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def confusion_grid(cms: Dict[str, np.ndarray], path) -> None:
    plt = _mpl()
    n = len(cms)
    cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (name, cm) in enumerate(cms.items()):
        ax = axes[i // cols][i % cols]; ax.axis("on")
        cm = np.asarray(cm, dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["W", "BC", "ST"], fontsize=7)
        ax.set_yticklabels(["W", "BC", "ST"], fontsize=7)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=8,
                        color="black" if norm[r, c] < 0.5 else "white")
        ax.set_title(name, fontsize=8)
        ax.set_ylabel("true", fontsize=7); ax.set_xlabel("pred", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def gate_inspection(gate_by_slice: Dict[str, np.ndarray], modality_names: List[str], path) -> None:
    """
    gate_by_slice[slice_label] = mean gate vector [M] for the full system within
    that stratum slice. Grouped bar chart (one group per slice, one bar per modality).
    """
    plt = _mpl()
    slices = list(gate_by_slice.keys())
    M = len(modality_names)
    x = np.arange(len(slices))
    width = 0.8 / max(M, 1)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(slices)), 3.6))
    for mi, mod in enumerate(modality_names):
        vals = [gate_by_slice[s][mi] for s in slices]
        ax.bar(x + mi * width, vals, width, label=mod, color=_COLORS[mi % len(_COLORS)])
    ax.set_xticks(x + width * (M - 1) / 2)
    ax.set_xticklabels(slices, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("mean modality gate")
    ax.set_title("Gate inspection (full system) by stratum slice")
    ax.legend(fontsize=7, frameon=False, ncol=M)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
