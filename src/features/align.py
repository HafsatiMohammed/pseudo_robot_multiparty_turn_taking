"""
Alignment of encoder outputs to the canonical 50 ms / 120-frame grid.

The grid for one sample is the 6 s context window [context_start, context_end=t]
divided into num_bins (120) bins of bin_dur (0.05) seconds. All cached modalities
are emitted on this grid so they stack frame-for-frame with X_frame [120, 7].
"""

from __future__ import annotations

import numpy as np


def pool_frames_to_grid(
    feats: np.ndarray,
    frame_times: np.ndarray,
    num_bins: int = 120,
    bin_dur: float = 0.05,
) -> np.ndarray:
    """
    Mean-pool encoder frames into fixed time bins.

    Args:
        feats: [T, ...] encoder-frame features (e.g. WavLM [T, L, D]).
        frame_times: [T] center time (s) of each frame, RELATIVE to the clip
            start (clip start == grid start == 0).
        num_bins: number of output bins (120).
        bin_dur: bin width in seconds (0.05).

    Returns:
        [num_bins, ...] pooled features; bins with no frame are zero.
    """
    feats = np.asarray(feats)
    out_shape = (num_bins,) + feats.shape[1:]
    out = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int64)

    bin_idx = np.floor(frame_times / bin_dur).astype(int)
    for i, b in enumerate(bin_idx):
        if 0 <= b < num_bins:
            out[b] += feats[i]
            counts[b] += 1
    nz = counts > 0
    out[nz] /= counts[nz].reshape((-1,) + (1,) * (feats.ndim - 1))
    return out


def place_words_on_grid(
    word_vecs: np.ndarray,
    word_spans: list[tuple[float, float]],
    clip_start: float,
    num_bins: int = 120,
    bin_dur: float = 0.05,
) -> np.ndarray:
    """
    Place per-word embeddings onto the grid by their time spans.

    Each word's vector is written to every bin its [start, end] overlaps
    (relative to clip_start). Bins covered by several words are averaged.
    Bins with no active word stay zero (preserves recency / silence structure).

    Args:
        word_vecs: [W, D] one vector per word.
        word_spans: list of (start, end) absolute times, len W.
        clip_start: grid start time (context_start) in seconds.
        num_bins, bin_dur: grid definition.

    Returns:
        [num_bins, D] grid-aligned embeddings.
    """
    if len(word_vecs) == 0:
        # D unknown when there are no words; caller passes a (0, D) array
        D = word_vecs.shape[1] if word_vecs.ndim == 2 else 0
        return np.zeros((num_bins, D), dtype=np.float32)

    D = word_vecs.shape[1]
    out = np.zeros((num_bins, D), dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int64)

    for vec, (s, e) in zip(word_vecs, word_spans):
        b0 = int(np.floor((s - clip_start) / bin_dur))
        b1 = int(np.floor((e - clip_start) / bin_dur))
        b0 = max(0, b0)
        b1 = min(num_bins - 1, b1)
        if b1 < b0:
            continue
        for b in range(b0, b1 + 1):
            out[b] += vec
            counts[b] += 1
    nz = counts > 0
    out[nz] /= counts[nz].reshape(-1, 1)
    return out


def wavlm_frame_times(num_frames: int, stride: float = 0.02, offset: float = 0.01) -> np.ndarray:
    """Center time (s) of each WavLM frame relative to the clip start.

    WavLM-base downsamples 16 kHz by 320 -> 50 Hz (20 ms hop). Frame i center is
    approximated as i*stride + offset."""
    return np.arange(num_frames) * stride + offset
