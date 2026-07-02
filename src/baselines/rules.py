"""
Rule baselines: Majority class, VA-Silence, VA-Threshold.

3-class predictions (idx WAIT=0, BACKCHANNEL=1, START_SPEAKING=2):
  VA-Silence  : START if silence_duration_before_t >= theta_start, else WAIT.
                (never predicts BC)
  VA-Threshold: BACKCHANNEL if (human active & exactly 1 speaker & no overlap &
                current_human_speech_duration >= theta_bc); START if
                silence >= theta_start; else WAIT. (silence/START and the BC
                condition are effectively mutually exclusive -- BC requires a
                live speaker, START requires silence.)

Thresholds are tuned on the VALIDATION split (macro-F1) and evaluated once on test.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..eval.metrics import BACKCHANNEL, START, WAIT, macro_f1

LABEL_TO_IDX = {"WAIT": 0, "BACKCHANNEL": 1, "START_SPEAKING": 2}


def labels_to_idx(df: pd.DataFrame, label_col: str = "final_label") -> np.ndarray:
    return df[label_col].map(LABEL_TO_IDX).to_numpy()


def majority_class(train_idx: np.ndarray) -> int:
    return int(np.bincount(train_idx, minlength=3).argmax())


def predict_majority(n: int, cls: int) -> np.ndarray:
    return np.full(n, cls, dtype=int)


def _bc_qualifies(df: pd.DataFrame, theta_bc: float) -> np.ndarray:
    return (
        df["human_active_at_t"].to_numpy(dtype=bool)
        & (df["num_humans_active_at_t"].to_numpy() == 1)
        & (~df["overlap_active_at_t"].to_numpy(dtype=bool))
        & (df["current_human_speech_duration"].to_numpy(dtype=float) >= theta_bc)
    )


def predict_va_silence(df: pd.DataFrame, theta_start: float) -> np.ndarray:
    sil = df["silence_duration_before_t"].to_numpy(dtype=float)
    pred = np.full(len(df), WAIT, dtype=int)
    pred[sil >= theta_start] = START
    return pred


def predict_va_threshold(df: pd.DataFrame, theta_start: float, theta_bc: float) -> np.ndarray:
    sil = df["silence_duration_before_t"].to_numpy(dtype=float)
    pred = np.full(len(df), WAIT, dtype=int)
    pred[_bc_qualifies(df, theta_bc)] = BACKCHANNEL
    pred[sil >= theta_start] = START  # START takes precedence (no conflict in practice)
    return pred


def tune_va_silence(
    val_df: pd.DataFrame, val_true: np.ndarray, grid: np.ndarray, progress: bool = False
) -> Tuple[float, float]:
    """Returns (best_macro_f1, best_theta_start)."""
    best_f1, best_th = -1.0, float(grid[0])
    it = grid
    if progress:
        import sys as _sys

        from tqdm import tqdm
        it = tqdm(grid, desc="tune:VA-Silence", dynamic_ncols=True, leave=False,
                  disable=not _sys.stderr.isatty())
    for th in it:
        f1 = macro_f1(val_true, predict_va_silence(val_df, th))
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_f1, best_th


def tune_va_threshold(
    val_df: pd.DataFrame, val_true: np.ndarray, grid_start: np.ndarray, grid_bc: np.ndarray,
    progress: bool = False,
) -> Tuple[float, float, float]:
    """Returns (best_macro_f1, best_theta_start, best_theta_bc)."""
    best = (-1.0, float(grid_start[0]), float(grid_bc[0]))
    it = grid_start
    if progress:
        import sys as _sys

        from tqdm import tqdm
        it = tqdm(grid_start, desc="tune:VA-Threshold", dynamic_ncols=True, leave=False,
                  disable=not _sys.stderr.isatty())
    for ts in it:
        for tb in grid_bc:
            f1 = macro_f1(val_true, predict_va_threshold(val_df, ts, tb))
            if f1 > best[0]:
                best = (f1, float(ts), float(tb))
    return best
