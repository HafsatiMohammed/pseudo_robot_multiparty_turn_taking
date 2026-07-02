"""
Fixes and additions for the VA rule baselines.

Contains:
  1. validate_va_features(df)  - corrected feature validation (replaces the buggy
                                 substring-based casting; hard-fails instead of
                                 silently defaulting).
  2. assert_labels_clean(df)   - guards against NaN labels after mapping.
  3. compute_det_eer(...)      - sweeps theta_start to trace the WAIT-vs-entry DET
                                 curve and return the interpolated EER.
  4. run_baseline_det(...)     - runs the sweep for both rule baselines on the test
                                 split, saves curves + EER, optional figure.

Metric conventions follow the metric sheet:
  miss rate        = count(pred=WAIT  & true in entry) / count(true in entry)
  false-alarm rate = count(pred=entry & true=WAIT)     / count(true=WAIT)
  EER              = the point where miss rate == false-alarm rate.
Both axes are normalised by GROUND TRUTH (detection convention) - note this
differs from the operating-point false-entry rate, which divides by predicted
entries.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENTRY_LABELS = {"BACKCHANNEL", "START_SPEAKING"}
VALID_LABELS = {"WAIT", "BACKCHANNEL", "START_SPEAKING"}

BOOL_FEATURES = ("human_active_at_t", "overlap_active_at_t")
INT_FEATURES = ("num_humans_active_at_t",)
FLOAT_FEATURES = ("silence_duration_before_t", "current_human_speech_duration")
REQUIRED_FEATURES = BOOL_FEATURES + INT_FEATURES + FLOAT_FEATURES


# ---------------------------------------------------------------------------
# 1. Corrected feature validation
# ---------------------------------------------------------------------------
def validate_va_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and type-cast the five voice-activity features.

    Fixes vs the original:
      - casts by an EXPLICIT column list, not a substring match. The old code did
        `if "active" in feat: astype(bool)`, which caught "num_humans_active_at_t"
        and turned the speaker COUNT into a bool -> the rule's `== 1` test became
        `True == 1`, firing for any number of active speakers.
      - HARD-FAILS on missing features instead of filling defaults (a defaulted
        baseline runs to completion and silently produces meaningless numbers).
      - rejects NaNs.
    """
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing voice-activity features: {missing}. "
            "Compute them from the AMI speech_regions (oracle voice activity); "
            "do not run on a VAD and do not default-fill for a paper baseline."
        )

    for feat in REQUIRED_FEATURES:
        if df[feat].isna().any():
            n = int(df[feat].isna().sum())
            raise ValueError(f"{n} NaN value(s) in feature '{feat}'.")

    for feat in BOOL_FEATURES:
        df[feat] = df[feat].astype(bool)
    for feat in INT_FEATURES:
        df[feat] = df[feat].round().astype(int)
    for feat in FLOAT_FEATURES:
        df[feat] = df[feat].astype(float)

    # sanity checks
    if (df["num_humans_active_at_t"] < 0).any():
        raise ValueError("Negative values in num_humans_active_at_t.")
    if (df["silence_duration_before_t"] < 0).any():
        raise ValueError("Negative values in silence_duration_before_t.")

    # cross-check overlap vs count (sources may differ; warn, don't fail)
    bad = df["overlap_active_at_t"] & (df["num_humans_active_at_t"] < 2)
    if bad.any():
        logger.warning(
            "%d row(s): overlap_active_at_t=True but <2 speakers active "
            "(inconsistent activity source).",
            int(bad.sum()),
        )
    # if a human is active, silence-before should be 0
    bad2 = df["human_active_at_t"] & (df["silence_duration_before_t"] > 0)
    if bad2.any():
        logger.warning(
            "%d row(s): human active but silence_duration_before_t > 0.",
            int(bad2.sum()),
        )
    return df


# ---------------------------------------------------------------------------
# 2. Label guard
# ---------------------------------------------------------------------------
def assert_labels_clean(df: pd.DataFrame, label_col: str = "final_label") -> None:
    """Fail loudly if labels contain anything outside the three classes."""
    bad = set(df[label_col].dropna().unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"Unexpected labels (not in {VALID_LABELS}): {bad}")
    if df[label_col].isna().any():
        raise ValueError(f"{int(df[label_col].isna().sum())} NaN label(s).")


# ---------------------------------------------------------------------------
# 3. DET / EER sweep
# ---------------------------------------------------------------------------
def compute_det_eer(
    df: pd.DataFrame,
    baseline: str = "silence",
    theta_bc: float = 1.0,
    n_grid: int = 400,
    label_col: str = "final_label",
) -> dict:
    """
    Sweep theta_start to trace the WAIT-vs-entry DET curve for a rule baseline.

    The binary decision is: act (pred in entry) vs WAIT.
      VA-Silence  : act  iff  silence >= theta_start
      VA-Threshold: act  iff  silence >= theta_start
                            OR (single speaker & no overlap & speech >= theta_bc)
    For VA-Threshold the BC rule acts independently of theta_start, so it offsets
    the curve (miss won't reach 1, false-alarm won't reach 0). That is expected
    and is itself informative.

    Returns dict with theta_grid, miss_rate, false_alarm_rate, eer, eer_theta.
    """
    if baseline not in {"silence", "threshold"}:
        raise ValueError("baseline must be 'silence' or 'threshold'.")

    true_entry = df[label_col].isin(ENTRY_LABELS).to_numpy()
    true_wait = ~true_entry
    n_entry, n_wait = int(true_entry.sum()), int(true_wait.sum())
    if n_entry == 0 or n_wait == 0:
        raise ValueError(
            f"Need both classes for a DET curve (entry={n_entry}, wait={n_wait})."
        )

    silence = df["silence_duration_before_t"].to_numpy(dtype=float)

    if baseline == "threshold":
        bc_qual = (
            df["human_active_at_t"].to_numpy(dtype=bool)
            & (df["num_humans_active_at_t"].to_numpy() == 1)
            & (~df["overlap_active_at_t"].to_numpy(dtype=bool))
            & (df["current_human_speech_duration"].to_numpy(dtype=float) >= theta_bc)
        )
    else:
        bc_qual = np.zeros(len(df), dtype=bool)

    hi = float(np.nanmax(silence)) + 0.1 if len(silence) else 1.0
    theta_grid = np.linspace(0.0, hi, n_grid)

    miss = np.empty(n_grid)
    fa = np.empty(n_grid)
    for i, th in enumerate(theta_grid):
        pred_entry = (silence >= th) | bc_qual
        miss[i] = np.sum(~pred_entry & true_entry) / n_entry
        fa[i] = np.sum(pred_entry & true_wait) / n_wait

    eer, eer_theta = _interp_eer(theta_grid, miss, fa)
    return {
        "baseline": baseline,
        "theta_grid": theta_grid,
        "miss_rate": miss,
        "false_alarm_rate": fa,
        "eer": eer,
        "eer_theta": eer_theta,
        "n_entry": n_entry,
        "n_wait": n_wait,
    }


def _interp_eer(theta: np.ndarray, miss: np.ndarray, fa: np.ndarray):
    """
    EER = crossing of miss and false-alarm. miss is non-decreasing in theta and
    fa non-increasing, so diff = miss - fa crosses zero once. Linear-interpolate.
    """
    diff = miss - fa
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        k = int(np.argmin(np.abs(diff)))
        logger.warning("No exact EER crossing on grid; reporting nearest point.")
        return float((miss[k] + fa[k]) / 2.0), float(theta[k])
    i = sign_change[0]
    d0, d1 = diff[i], diff[i + 1]
    t = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    miss_e = miss[i] + t * (miss[i + 1] - miss[i])
    fa_e = fa[i] + t * (fa[i + 1] - fa[i])
    theta_e = theta[i] + t * (theta[i + 1] - theta[i])
    return float((miss_e + fa_e) / 2.0), float(theta_e)


# ---------------------------------------------------------------------------
# 4. Runner: both baselines on the test split
# ---------------------------------------------------------------------------
def run_baseline_det(
    df: pd.DataFrame,
    output_dir: str,
    theta_bc: float = 1.0,
    split: str = "test",
    label_col: str = "final_label",
    make_figure: bool = True,
) -> dict:
    """
    Run the DET/EER sweep for VA-Silence and VA-Threshold on the test split
    (the natural distribution), save per-curve CSVs and a combined EER JSON, and
    optionally a DET figure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "split" in df.columns and split in set(df["split"].unique()):
        eval_df = df[df["split"] == split].copy()
    else:
        logger.warning("Split '%s' not found; using all rows for DET.", split)
        eval_df = df.copy()

    eval_df = validate_va_features(eval_df)
    assert_labels_clean(eval_df, label_col)

    results = {}
    for name in ("silence", "threshold"):
        r = compute_det_eer(eval_df, baseline=name, theta_bc=theta_bc, label_col=label_col)
        pd.DataFrame(
            {
                "theta_start": r["theta_grid"],
                "miss_rate": r["miss_rate"],
                "false_alarm_rate": r["false_alarm_rate"],
            }
        ).to_csv(out / f"det_curve_va_{name}.csv", index=False)
        results[name] = {
            "eer": r["eer"],
            "eer_theta": r["eer_theta"],
            "n_entry": r["n_entry"],
            "n_wait": r["n_wait"],
        }
        logger.info("VA-%s EER = %.4f at theta_start = %.3f", name, r["eer"], r["eer_theta"])

    import json

    with open(out / "det_eer.json", "w") as f:
        json.dump(results, f, indent=2)

    if make_figure:
        _plot_det(eval_df, theta_bc, out / "det_curve.png", label_col)

    return results


def _plot_det(df, theta_bc, path, label_col):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping DET figure.")
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for name, color in (("silence", "#D85A30"), ("threshold", "#1D9E75")):
        r = compute_det_eer(df, baseline=name, theta_bc=theta_bc, label_col=label_col)
        ax.plot(r["false_alarm_rate"], r["miss_rate"], color=color,
                label=f"VA-{name} (EER={r['eer']:.2f})")
        ax.plot(r["eer"], r["eer"], "o", color=color, ms=6)
    ax.plot([0, 1], [0, 1], "--", color="#888780", lw=1, label="EER line")
    ax.set_xlabel("false-alarm rate (of true WAITs)")
    ax.set_ylabel("miss rate (of true entries)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved DET figure: %s", path)
