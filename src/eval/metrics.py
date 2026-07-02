"""
Phase 6: evaluation metrics — implements EXACTLY icassp_metric_definitions.md.

Label space: WAIT=0, BACKCHANNEL=1, START_SPEAKING=2.
entry = {BACKCHANNEL, START_SPEAKING}.

Two regimes kept distinct:
  - operating-point metrics (argmax preds): per-class P/R/F1, macro/weighted-F1,
    balanced accuracy, false-entry, missed-entry, action-type errors, ECE.
  - swept metric: DET curve + EER on the binary WAIT-vs-entry decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

WAIT = 0
BACKCHANNEL = 1
START = 2
ENTRY = (BACKCHANNEL, START)
LABELS = ("WAIT", "BACKCHANNEL", "START_SPEAKING")
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Per-class P/R/F1 and aggregates
# ---------------------------------------------------------------------------
def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        pred_c = int(np.sum(y_pred == c))
        true_c = int(np.sum(y_true == c))
        precision = tp / pred_c if pred_c > 0 else 0.0
        recall = tp / true_c if true_c > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[LABELS[c]] = {"precision": precision, "recall": recall, "f1": f1, "support": true_c}
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    prf = per_class_prf(y_true, y_pred, num_classes)
    return float(np.mean([prf[LABELS[c]]["f1"] for c in range(num_classes)]))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    prf = per_class_prf(y_true, y_pred, num_classes)
    total = len(y_true)
    if total == 0:
        return 0.0
    return float(sum(prf[LABELS[c]]["f1"] * prf[LABELS[c]]["support"] for c in range(num_classes)) / total)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    prf = per_class_prf(y_true, y_pred, num_classes)
    recalls = [prf[LABELS[c]]["recall"] for c in range(num_classes) if prf[LABELS[c]]["support"] > 0]
    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# Deployment-critical act-or-wait metrics (operating point)
# ---------------------------------------------------------------------------
def false_entry_rate(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """count(pred in entry & true=WAIT) / count(pred in entry); None if denom 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_entry = np.isin(y_pred, ENTRY)
    denom = int(pred_entry.sum())
    if denom == 0:
        return None
    num = int(np.sum(pred_entry & (y_true == WAIT)))
    return num / denom


def missed_entry_rate(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """count(pred=WAIT & true in entry) / count(true in entry); None if denom 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_entry = np.isin(y_true, ENTRY)
    denom = int(true_entry.sum())
    if denom == 0:
        return None
    num = int(np.sum((y_pred == WAIT) & true_entry))
    return num / denom


def action_type_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """BC-as-turn = count(true=BC & pred=START)/count(true=BC); turn-as-BC symmetric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_bc = int(np.sum(y_true == BACKCHANNEL))
    n_st = int(np.sum(y_true == START))
    bc_as_turn = (int(np.sum((y_true == BACKCHANNEL) & (y_pred == START))) / n_bc) if n_bc else None
    turn_as_bc = (int(np.sum((y_true == START) & (y_pred == BACKCHANNEL))) / n_st) if n_st else None
    return {"bc_as_turn": bc_as_turn, "turn_as_bc": turn_as_bc}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# DET / EER (swept; WAIT-vs-entry). Both axes truth-normalized.
# ---------------------------------------------------------------------------
def _interp_eer(theta: np.ndarray, miss: np.ndarray, fa: np.ndarray):
    """EER = crossing of miss and fa (miss non-decreasing, fa non-increasing in theta)."""
    diff = miss - fa
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        k = int(np.argmin(np.abs(diff)))
        return float((miss[k] + fa[k]) / 2.0), float(theta[k])
    i = sign_change[0]
    d0, d1 = diff[i], diff[i + 1]
    tt = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    miss_e = miss[i] + tt * (miss[i + 1] - miss[i])
    fa_e = fa[i] + tt * (fa[i + 1] - fa[i])
    theta_e = theta[i] + tt * (theta[i + 1] - theta[i])
    return float((miss_e + fa_e) / 2.0), float(theta_e)


def det_eer_from_scores(y_true: np.ndarray, scores: np.ndarray, n_grid: int = 512) -> Dict:
    """
    Sweep an entry score (= 1 - P(WAIT) for trained heads). Decision: act iff
    score >= tau. Returns the DET curve and interpolated EER.
      miss(tau)        = count(pred=WAIT  & true in entry) / count(true in entry)
      false_alarm(tau) = count(pred=entry & true=WAIT)     / count(true=WAIT)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    true_entry = np.isin(y_true, ENTRY)
    true_wait = ~true_entry
    n_entry, n_wait = int(true_entry.sum()), int(true_wait.sum())
    if n_entry == 0 or n_wait == 0:
        raise ValueError(f"DET needs both classes (entry={n_entry}, wait={n_wait}).")

    lo = float(min(0.0, scores.min()))
    hi = float(max(1.0, scores.max())) + 1e-6
    theta = np.linspace(lo, hi, n_grid)
    miss = np.empty(n_grid)
    fa = np.empty(n_grid)
    for i, th in enumerate(theta):
        pred_entry = scores >= th
        miss[i] = np.sum(~pred_entry & true_entry) / n_entry
        fa[i] = np.sum(pred_entry & true_wait) / n_wait
    eer, eer_theta = _interp_eer(theta, miss, fa)
    return {
        "theta": theta, "miss_rate": miss, "false_alarm_rate": fa,
        "eer": eer, "eer_theta": eer_theta, "n_entry": n_entry, "n_wait": n_wait,
    }


def entry_score_from_probs(probs: np.ndarray) -> np.ndarray:
    """score = 1 - P(WAIT)."""
    probs = np.asarray(probs, dtype=float)
    return 1.0 - probs[:, WAIT]


# ---------------------------------------------------------------------------
# Calibration (ECE)
# ---------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Bootstrap significance for macro-F1 difference (full vs best single-modality)
# ---------------------------------------------------------------------------
def bootstrap_macro_f1_diff(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    reps: int = 1000, seed: int = 42, ci: float = 0.95, progress: bool = False,
) -> Dict:
    """Paired bootstrap of macroF1(a) - macroF1(b) over resampled test indices."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    rep_iter = range(reps)
    if progress:
        import sys as _sys

        from tqdm import tqdm
        rep_iter = tqdm(rep_iter, desc="bootstrap", dynamic_ncols=True, leave=False,
                        disable=not _sys.stderr.isatty())
    for r in rep_iter:
        idx = rng.integers(0, n, size=n)
        diffs[r] = macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    observed = macro_f1(y_true, pred_a) - macro_f1(y_true, pred_b)
    return {
        "observed_diff": float(observed),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo), "ci_high": float(hi), "ci": ci,
        "frac_gt_0": float(np.mean(diffs > 0)), "reps": reps,
    }


# ---------------------------------------------------------------------------
# Assemble the operating-point suite (+ EER if probabilities are given)
# ---------------------------------------------------------------------------
def compute_all(y_true: np.ndarray, y_pred: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict:
    prf = per_class_prf(y_true, y_pred)
    res = {
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "wait_f1": prf["WAIT"]["f1"],
        "bc_f1": prf["BACKCHANNEL"]["f1"],
        "start_f1": prf["START_SPEAKING"]["f1"],
        "false_entry": false_entry_rate(y_true, y_pred),
        "missed_entry": missed_entry_rate(y_true, y_pred),
        "per_class": prf,
        **action_type_errors(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
    }
    if probs is not None:
        scores = entry_score_from_probs(probs)
        try:
            det = det_eer_from_scores(y_true, scores)
            res["eer"] = det["eer"]
            res["eer_theta"] = det["eer_theta"]
        except ValueError:
            res["eer"] = None
        res["ece"] = expected_calibration_error(probs, y_true)
    return res
