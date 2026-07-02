"""
Timing feature extraction (Phase 1a).

Produces, per sample, strictly from word activity with timestamp < t:
  - X_frame  [120, 7]   per-50ms-frame group-activity features
  - X_scalar [6]        scalar features at the decision point t
  - va        dict      5 oracle voice-activity features for the rule baselines

Definitions follow docs/DATA_SCHEMA.md (which transcribes icassp_phase0_scope.md
and the prepare_dataset.py draft). "Humans" = all meeting agents except the
pseudo_robot; the normaliser for num_humans_active_norm is the human count
(group_size - 1), which varies by meeting.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .regions import (
    Region,
    active_letters_at,
    covers,
    last_offset_before,
    ongoing_run_onset,
    time_with_min_speakers,
)

FRAME_FEATURES = [
    "any_human_active",          # 0
    "num_humans_active_norm",    # 1
    "overlap_active",            # 2
    "silence_active",            # 3
    "human_onset",               # 4
    "human_offset",              # 5
    "pseudo_robot_past_active",  # 6
]

SCALAR_FEATURES = [
    "silence_duration_before_t",           # 0
    "current_human_speech_duration",       # 1
    "human_speech_ratio_last_1s",          # 2
    "human_speech_ratio_last_6s",          # 3
    "overlap_ratio_last_6s",               # 4
    "time_since_pseudo_robot_last_spoke",  # 5
]

VA_FEATURES = [
    "human_active_at_t",
    "num_humans_active_at_t",
    "overlap_active_at_t",
    "silence_duration_before_t",
    "current_human_speech_duration",
]

DUR_CAP = 10.0  # seconds, matches the prepare_dataset draft


def extract_frame_features(
    human_regions: Dict[str, List[Region]],
    robot_regions: List[Region],
    context_start: float,
    num_frames: int = 120,
    frame_shift: float = 0.05,
) -> np.ndarray:
    """X_frame [num_frames, 7]. Frame i is the instant context_start + i*shift."""
    num_humans = len(human_regions)
    frame_times = context_start + np.arange(num_frames) * frame_shift
    x = np.zeros((num_frames, 7), dtype=np.float32)

    prev_nh = 0
    for i, tau in enumerate(frame_times):
        active_h = active_letters_at(human_regions, tau)
        nh = len(active_h)
        robot_act = covers(robot_regions, tau)

        x[i, 0] = 1.0 if nh > 0 else 0.0
        x[i, 1] = (nh / num_humans) if num_humans > 0 else 0.0
        x[i, 2] = 1.0 if nh > 1 else 0.0
        x[i, 3] = 1.0 if (nh == 0 and not robot_act) else 0.0
        if i > 0:
            x[i, 4] = 1.0 if nh > prev_nh else 0.0
            x[i, 5] = 1.0 if nh < prev_nh else 0.0
        x[i, 6] = 1.0 if robot_act else 0.0
        prev_nh = nh
    return x


def extract_scalar_features(
    human_regions: Dict[str, List[Region]],
    robot_regions: List[Region],
    t: float,
    context_seconds: float = 6.0,
) -> np.ndarray:
    """X_scalar [6] at the decision point t."""
    x = np.zeros(6, dtype=np.float32)

    active_h = active_letters_at(human_regions, t)
    human_active = len(active_h) > 0

    # [0] silence since any human last active
    if human_active:
        x[0] = 0.0
    else:
        last_ends = [
            last_offset_before(regs, t) for regs in human_regions.values()
        ]
        last_ends = [e for e in last_ends if e is not None]
        x[0] = min(t - max(last_ends), DUR_CAP) if last_ends else DUR_CAP

    # [1] current (ongoing) human speech-run duration
    if human_active:
        onsets = [
            ongoing_run_onset(human_regions[h], t) for h in active_h
        ]
        onsets = [o for o in onsets if o is not None]
        x[1] = min(t - min(onsets), DUR_CAP) if onsets else 0.0
    else:
        x[1] = 0.0

    # [2]/[3] fraction of last 1s / 6s with >=1 human active
    x[2] = time_with_min_speakers(human_regions, t - 1.0, t, 1) / 1.0
    x[3] = time_with_min_speakers(human_regions, t - context_seconds, t, 1) / context_seconds

    # [4] fraction of last 6s with >=2 humans active
    x[4] = time_with_min_speakers(human_regions, t - context_seconds, t, 2) / context_seconds

    # [5] silence since pseudo-robot last active
    if covers(robot_regions, t):
        x[5] = 0.0
    else:
        last = last_offset_before(robot_regions, t)
        x[5] = min(t - last, DUR_CAP) if last is not None else DUR_CAP

    return x


def extract_va_features(
    human_regions: Dict[str, List[Region]],
    x_scalar: np.ndarray,
    t: float,
) -> Dict[str, float]:
    """5 oracle voice-activity features for the rule baselines (at t)."""
    active_h = active_letters_at(human_regions, t)
    n = len(active_h)
    return {
        "human_active_at_t": bool(n > 0),
        "num_humans_active_at_t": int(n),
        "overlap_active_at_t": bool(n > 1),
        "silence_duration_before_t": float(x_scalar[0]),
        "current_human_speech_duration": float(x_scalar[1]),
    }


def extract_sample(
    human_regions: Dict[str, List[Region]],
    robot_regions: List[Region],
    t: float,
    context_start: float,
    num_frames: int = 120,
    frame_shift: float = 0.05,
    context_seconds: float = 6.0,
):
    """Convenience: returns (X_frame[120,7], X_scalar[6], va_dict)."""
    x_frame = extract_frame_features(
        human_regions, robot_regions, context_start, num_frames, frame_shift
    )
    x_scalar = extract_scalar_features(
        human_regions, robot_regions, t, context_seconds
    )
    va = extract_va_features(human_regions, x_scalar, t)
    return x_frame, x_scalar, va
