"""
Phase 7: assemble the main + stratified result tables (Markdown + LaTeX booktabs).

mean +/- std over seeds for trained rows; single value for non-learned baselines.
Best per column is bold (max for F1, min for false-entry/missed-entry/EER).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# (column key, header, higher_is_better)
MAIN_COLS: List[Tuple[str, str, bool]] = [
    ("macro_f1", "Macro-F1", True),
    ("wait_f1", "WAIT F1", True),
    ("bc_f1", "BC F1", True),
    ("start_f1", "START F1", True),
    ("false_entry", "False-entry", False),
    ("missed_entry", "Missed-entry", False),
    ("eer", "EER", False),
]

# (row key, display name) in the paper's order
MAIN_ROWS: List[Tuple[str, str]] = [
    ("majority", "Majority class"),
    ("va_silence", "VA-Silence"),
    ("va_threshold", "VA-Threshold"),
    ("timing", "Timing-only"),
    ("audio_timing", "Audio + timing"),
    ("text_timing", "Text + timing"),
    ("full", "GroupTurn-Fuse (full)"),
]
TRAINED = {"timing", "audio_timing", "text_timing", "full"}


def _agg(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """mean, std over non-None values; (None, None) if all None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def aggregate(
    baseline_metrics: Dict, trained: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """
    baseline_metrics: {row_key: {"test": {...}}} for majority/va_silence/va_threshold.
    trained: {system: [final_metrics_per_seed, ...]} where each has ["test"][col].
    Returns {row_key: {col: (mean, std)}}.
    """
    out: Dict[str, Dict] = {}
    for key, _name in MAIN_ROWS:
        out[key] = {}
        for col, _h, _ in MAIN_COLS:
            if key in TRAINED:
                vals = [fm["test"].get(col) for fm in trained.get(key, [])]
            else:
                t = baseline_metrics.get(key, {}).get("test", {})
                vals = [t.get(col)]
            out[key][col] = _agg(vals)
    return out


def _best_per_col(agg: Dict) -> Dict[str, str]:
    """Row key holding the best mean for each column."""
    best = {}
    for col, _h, higher in MAIN_COLS:
        candidates = [(k, agg[k][col][0]) for k, _ in MAIN_ROWS if agg[k][col][0] is not None]
        if not candidates:
            continue
        best[col] = (max if higher else min)(candidates, key=lambda kv: kv[1])[0]
    return best


def _fmt(mean: Optional[float], std: Optional[float], bold: bool, latex: bool) -> str:
    if mean is None:
        return "--"
    s = f"{mean:.3f}" if (std is None or std == 0.0) else f"{mean:.3f} ± {std:.3f}"
    if std is not None and std > 0 and latex:
        s = f"{mean:.3f} $\\pm$ {std:.3f}"
    if bold:
        s = (f"\\textbf{{{s}}}" if latex else f"**{s}**")
    return s


def render_main_markdown(agg: Dict) -> str:
    best = _best_per_col(agg)
    head = "| System | " + " | ".join(h for _, h, _ in MAIN_COLS) + " |"
    sep = "|" + "---|" * (len(MAIN_COLS) + 1)
    lines = ["# Main results (test split; natural distribution)\n", head, sep]
    for key, name in MAIN_ROWS:
        cells = []
        for col, _h, _ in MAIN_COLS:
            mean, std = agg[key][col]
            cells.append(_fmt(mean, std, bold=(best.get(col) == key), latex=False))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_main_latex(agg: Dict) -> str:
    best = _best_per_col(agg)
    ncol = len(MAIN_COLS)
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Turn-taking results on the natural (WAIT-dominated) test "
        "distribution. Mean $\\pm$ std over seeds for trained rows; best per column bold.}",
        "\\label{tab:main}",
        "\\begin{tabular}{l" + "c" * ncol + "}",
        "\\toprule",
        "System & " + " & ".join(h for _, h, _ in MAIN_COLS) + " \\\\",
        "\\midrule",
    ]
    for ri, (key, name) in enumerate(MAIN_ROWS):
        cells = []
        for col, _h, _ in MAIN_COLS:
            mean, std = agg[key][col]
            cells.append(_fmt(mean, std, bold=(best.get(col) == key), latex=True))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if key == "va_threshold":
            lines.append("\\midrule")  # separate baselines from trained rows
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stratified tables (S1-S3): macro-F1 (or FE/ME for small cells) per slice/system
# ---------------------------------------------------------------------------
def render_stratified_markdown(strata: Dict, systems: List[str]) -> str:
    lines = ["# Stratified diagnostic (test split)\n"]
    for sid, S in strata.items():
        lines.append(f"## {sid} — {S['label']}\n")
        header = "| Slice | " + " | ".join(systems) + " | (Δ vs timing) |"
        lines.append(header)
        lines.append("|" + "---|" * (len(systems) + 2))
        for slabel, per_sys in S["slices"].items():
            cells = []
            n = None
            for name in systems:
                m = per_sys.get(name, {})
                n = m.get("n", n)
                if m.get("small_cell", True):
                    fe, me = m.get("false_entry"), m.get("missed_entry")
                    cells.append(f"FE={_n(fe)} ME={_n(me)}")
                else:
                    cells.append(f"{m.get('macro_f1', float('nan')):.3f}")
            deltas = S["delta_macro_f1"].get(slabel, {})
            dtxt = ", ".join(f"{k}:{v:+.3f}" for k, v in deltas.items() if k != "timing")
            lines.append(f"| {slabel} (n={n}) | " + " | ".join(cells) + f" | {dtxt} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _n(x):
    return "null" if x is None else f"{x:.3f}"
