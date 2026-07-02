#!/usr/bin/env python3
"""
Phase 4: non-learned baselines on the timing parquet.

  - Majority class (most frequent TRAIN label) -> predict on test.
  - VA-Silence / VA-Threshold: tune theta_start (and theta_bc) on VAL macro-F1,
    evaluate ONCE on test. Uses validate_va_features (hard-fails on missing
    features, fixes the count-vs-bool cast). DET/EER via run_baseline_det
    (sweeps theta_start; both axes truth-normalized; EER interpolated).

Evaluated on the natural (WAIT-dominated) test distribution. Tuning never touches
test. Writes a metrics JSON + a Markdown table + DET curves/figure under --output.

Usage:
    python scripts/eval_baselines.py --timing-dir data/processed/timing \
        --output reports/baselines
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baselines.rules import (
    labels_to_idx, majority_class, predict_majority,
    predict_va_silence, predict_va_threshold, tune_va_silence, tune_va_threshold,
)
from src.baselines.va_baseline_fixes import (
    assert_labels_clean, run_baseline_det, validate_va_features,
)
from src.eval.metrics import compute_all, macro_f1
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _fmt(x):
    return "null" if x is None else f"{x:.4f}"


def main(args):
    timing = Path(args.timing_dir)
    train = pd.read_parquet(timing / "train.parquet")
    val = pd.read_parquet(timing / "validation.parquet")
    test = pd.read_parquet(timing / "test.parquet")
    if args.max_samples:
        train, val, test = train[: args.max_samples], val[: args.max_samples], test[: args.max_samples]

    for df in (train, val, test):
        validate_va_features(df)
        assert_labels_clean(df)

    y_train = labels_to_idx(train)
    y_val = labels_to_idx(val)
    y_test = labels_to_idx(test)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, level=args.log_level, filename="baselines.log")
    logger.info("baselines | train=%d val=%d test=%d", len(train), len(val), len(test))
    results = {}
    test_ids = test["sample_id"].tolist()

    # --- Majority class ---
    cls = majority_class(y_train)
    pred = predict_majority(len(test), cls)
    results["majority"] = {
        "params": {"class": int(cls)},
        "test": compute_all(y_test, pred),  # majority has no probs -> no EER
    }

    # --- VA-Silence ---
    grid_start = np.round(np.arange(0.0, args.max_theta + 1e-9, args.theta_step), 4)
    f1_s, th_s = tune_va_silence(val, y_val, grid_start, progress=True)
    pred_s = predict_va_silence(test, th_s)
    results["va_silence"] = {
        "params": {"theta_start": th_s, "val_macro_f1": f1_s},
        "test": compute_all(y_test, pred_s),
    }

    # --- VA-Threshold ---
    grid_bc = np.round(np.arange(args.theta_step, args.max_theta + 1e-9, args.theta_step), 4)
    f1_t, th_st, th_bc = tune_va_threshold(val, y_val, grid_start, grid_bc, progress=True)
    pred_t = predict_va_threshold(test, th_st, th_bc)
    results["va_threshold"] = {
        "params": {"theta_start": th_st, "theta_bc": th_bc, "val_macro_f1": f1_t},
        "test": compute_all(y_test, pred_t),
    }

    # --- save test predictions (so the table/DET can be recomputed without re-tuning) ---
    np.savez(out / "baseline_preds_test.npz", sample_id=np.array(test_ids), y_true=y_test,
             majority=pred, va_silence=pred_s, va_threshold=pred_t)

    # --- DET / EER for both rules on test (sweeps theta_start) ---
    try:
        det = run_baseline_det(
            test, output_dir=str(out), theta_bc=th_bc, split="test", make_figure=True
        )
        results["va_silence"]["test"]["eer"] = det["silence"]["eer"]
        results["va_threshold"]["test"]["eer"] = det["threshold"]["eer"]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] DET sweep skipped: {e}")

    # --- save JSON ---
    with open(out / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # --- Markdown summary ---
    lines = [
        "# Baseline results (test split)\n",
        "| System | Macro-F1 | WAIT F1 | BC F1 | START F1 | False-entry | Missed-entry | EER |",
        "|---|---|---|---|---|---|---|---|",
    ]
    row_order = [("Majority class", "majority"), ("VA-Silence", "va_silence"), ("VA-Threshold", "va_threshold")]
    for label, key in row_order:
        t = results[key]["test"]
        lines.append(
            f"| {label} | {_fmt(t['macro_f1'])} | {_fmt(t['wait_f1'])} | {_fmt(t['bc_f1'])} | "
            f"{_fmt(t['start_f1'])} | {_fmt(t['false_entry'])} | {_fmt(t['missed_entry'])} | "
            f"{_fmt(t.get('eer'))} |"
        )
    (out / "baseline_table.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))  # readable preview to stdout
    logger.info("Tuned: VA-Silence theta_start=%s (val F1=%.3f); "
                "VA-Threshold theta_start=%s, theta_bc=%s (val F1=%.3f)",
                th_s, f1_s, th_st, th_bc, f1_t)
    logger.info("Saved -> %s/baseline_metrics.json, baseline_table.md, "
                "baseline_preds_test.npz, det_*.csv, det_curve.png", out)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--output", default="reports/baselines")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-theta", type=float, default=4.0, help="Max theta for grid sweep (s).")
    p.add_argument("--theta-step", type=float, default=0.1, help="Grid step (s).")
    p.add_argument("--log-level", default="INFO")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
