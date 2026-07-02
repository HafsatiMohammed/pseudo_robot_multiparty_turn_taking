#!/usr/bin/env python3
"""
Fixed WavLM layer-weighting tests (runnable: `python tests/test_audio_wavlm_weights.py`).

These cover the layer_mode="sum" collapse ONLY -- the numpy weight-vector logic and
the einsum contraction. They do NOT load WavLM/transformers, so they run in any env
with numpy.

1. build_layer_weights: default prior = layer 3-8 mean, normalized; explicit vectors
   are normalized; wrong length / non-positive sum raise; layer_mode="all" -> None.
2. The encode-time contraction np.einsum("l,tld->td", w, grid) equals a manual weighted
   sum and yields shape [T, D].
"""

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.features.audio_wavlm import build_layer_weights


def test_default_prior_is_layer_3_8_mean():
    w = build_layer_weights("sum", None, num_layers=13)
    assert w.shape == (13,)
    assert w.dtype == np.float32
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64)
    expected /= expected.sum()
    assert np.allclose(w, expected)
    assert np.isclose(w.sum(), 1.0)
    assert np.flatnonzero(w).tolist() == [3, 4, 5, 6, 7, 8]
    print("[ok] default prior = normalized layer 3-8 mean")


def test_explicit_weights_are_normalized():
    raw = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    w = build_layer_weights("sum", raw, num_layers=13)
    assert np.isclose(w.sum(), 1.0)
    # unequal explicit weights preserve their relative proportions after normalization
    raw2 = [0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w2 = build_layer_weights("sum", raw2, num_layers=13)
    assert np.isclose(w2[2] / w2[3], 2.0)
    assert np.isclose(w2.sum(), 1.0)
    print("[ok] explicit weights normalized to sum 1, proportions preserved")


def test_all_mode_returns_none():
    assert build_layer_weights("all", None, num_layers=13) is None
    assert build_layer_weights("all", [1] * 13, num_layers=13) is None
    print("[ok] layer_mode='all' -> None (no collapse)")


def test_invalid_inputs_raise():
    for bad in ([1, 1, 1], [0] * 14):  # wrong length
        try:
            build_layer_weights("sum", bad, num_layers=13)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for length {len(bad)}")
    try:
        build_layer_weights("sum", [0] * 13, num_layers=13)  # sums to zero
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for all-zero weights")
    print("[ok] wrong length / non-positive sum raise ValueError")


def test_einsum_contraction_matches_manual_and_shape():
    rng = np.random.default_rng(0)
    grid = rng.standard_normal((120, 13, 768)).astype(np.float32)  # [T, L, D]
    w = build_layer_weights("sum", None, num_layers=13)

    out = np.einsum("l,tld->td", w, grid).astype(np.float32)
    assert out.shape == (120, 768)

    # manual reference: weighted sum over the layer axis
    ref = (grid * w[None, :, None]).sum(axis=1).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-5)

    # with the layer 3-8 mean prior this equals the plain mean of layers 3-8
    ref_band = grid[:, 3:9, :].mean(axis=1).astype(np.float32)
    assert np.allclose(out, ref_band, atol=1e-5)
    print("[ok] einsum('l,tld->td') == manual weighted sum, shape [120,768]")


if __name__ == "__main__":
    test_default_prior_is_layer_3_8_mean()
    test_explicit_weights_are_normalized()
    test_all_mode_returns_none()
    test_invalid_inputs_raise()
    test_einsum_contraction_matches_manual_and_shape()
    print("\n*** WAVLM LAYER-WEIGHT TESTS PASSED ***")
