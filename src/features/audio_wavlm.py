"""
WavLM audio feature caching (frozen encoder, offline).

The encoder is loaded once in eval() with requires_grad_(False) and run under
no_grad. It is NEVER instantiated or backpropped during training -- features are
written to disk and the data layer reads them. (Non-negotiable: frozen encoders.)

Per sample:
  human-group-only audio = mean-mix of the OTHER speakers' headset WAVs over
  [context_start, context_end], 16 kHz mono, fixed to exactly 6 s (96000 samp).
  WavLM (~20 ms frames) -> mean-pool into the 120 x 50 ms bins.
  layer_mode="all" -> [120, L, D] (enables learnable layer-weighting)
  layer_mode="sum" -> [120, D]    (FIXED weighted layer combine; smaller on disk)

layer_mode="sum" does NOT flat-sum all 13 hidden states. It collapses the layer
axis with a fixed (not learned) normalized weight vector concentrated on the
lower-middle transformer layers (default: uniform mean of layers 3-8), which
prior layerwise analyses show carry the paralinguistic / turn-taking signal.
The weights are fixed on purpose so the offline cache stays a single [T, 768]
representation (~150 GB) instead of the ~13x-larger all-layers tensor. See the
paper note: "fixed WavLM layer-3-8 mean".
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import torch

from .align import pool_frames_to_grid, wavlm_frame_times

logger = logging.getLogger(__name__)

# Default fixed layer prior for layer_mode="sum": uniform mean over the
# lower-middle transformer layers (indices 3..8 inclusive), zero elsewhere.
# WavLM-base-plus exposes 13 hidden states (index 0 = CNN/embedding output,
# 1..12 = transformer layers).
_DEFAULT_SUM_BAND = (3, 8)  # inclusive layer indices


def build_layer_weights(
    layer_mode: str,
    layer_weights: Optional[Sequence[float]],
    num_layers: int,
) -> Optional[np.ndarray]:
    """Return the normalized [num_layers] weight vector used to collapse the WavLM
    layer axis when ``layer_mode == "sum"``, or ``None`` for ``layer_mode == "all"``.

    - ``layer_mode == "all"``: returns ``None`` (layer axis is kept; no collapse).
    - explicit ``layer_weights``: validated to length ``num_layers`` and normalized
      to sum to 1.
    - ``None`` + ``"sum"``: the documented default prior -- a uniform mean over
      layers 3..8 inclusive (zero elsewhere), normalized. This is NOT a flat mean
      over all layers.
    """
    if layer_mode != "sum":
        return None
    if layer_weights is not None:
        w = np.asarray(layer_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != num_layers:
            raise ValueError(
                f"layer_weights must have length {num_layers} (got {w.shape[0]}); "
                "WavLM-base-plus exposes 13 hidden states (0=CNN/embed, 1..12=transformer)."
            )
    else:
        lo, hi = _DEFAULT_SUM_BAND
        if hi >= num_layers:
            raise ValueError(
                f"default layer band {_DEFAULT_SUM_BAND} exceeds num_layers={num_layers}; "
                "pass an explicit layer_weights vector."
            )
        w = np.zeros(num_layers, dtype=np.float64)
        w[lo : hi + 1] = 1.0
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("layer_weights must sum to a positive value.")
    return (w / total).astype(np.float32)


class WavLMCacher:
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: str = "cpu",
        layer_mode: str = "all",
        sample_rate: int = 16000,
        layer_weights: Optional[Sequence[float]] = None,
    ):
        if layer_mode not in ("all", "sum"):
            raise ValueError("layer_mode must be 'all' or 'sum'")
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = device
        self.layer_mode = layer_mode
        self.sample_rate = sample_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        # Fixed (not learned) layer-collapse weights, only used for layer_mode="sum".
        # Logged once so each cache run records exactly which layers were used.
        self.layer_weights = build_layer_weights(layer_mode, layer_weights, self.num_layers)
        if self.layer_weights is not None:
            nz = np.flatnonzero(self.layer_weights).tolist()
            logger.info(
                "WavLM layer_mode=sum | FIXED normalized layer weights (len=%d): %s "
                "| nonzero layer indices=%s",
                len(self.layer_weights),
                np.array2string(self.layer_weights, precision=4, separator=",", max_line_width=200),
                nz,
            )

    def _grid_from_frames(self, feats: np.ndarray, num_bins: int, bin_dur: float) -> np.ndarray:
        """[T, L, D] float32 encoder frames -> [num_bins, L, D] (all) or [num_bins, D]
        (sum), as float16 for the on-disk cache. All pooling/weighting math stays
        float32; only the returned array is cast to float16."""
        T = feats.shape[0]
        ftimes = wavlm_frame_times(T)
        grid = pool_frames_to_grid(feats, ftimes, num_bins=num_bins, bin_dur=bin_dur)  # [120,L,D]
        if self.layer_mode == "sum":
            # Fixed weighted combination across the layer axis -> [120, D].
            # (Not a flat sum -- weights front-load the lower-middle layers.)
            grid = np.einsum("l,tld->td", self.layer_weights, grid).astype(np.float32)  # [120, D]
        # Frozen features feed a GRU; float16 on disk is plenty and halves cache size.
        return grid.astype(np.float16)

    @torch.no_grad()
    def encode_batch(
        self,
        waveforms: Sequence[np.ndarray],
        num_bins: int = 120,
        bin_dur: float = 0.05,
    ) -> List[np.ndarray]:
        """Batched form of ``encode``: one WavLM forward for B clips.

        waveforms: sequence of 1-D float32 @ sample_rate. In this pipeline every clip
        is fixed to exactly context_seconds (96000 samp), so the batch needs no padding
        and each sample's result is identical to encoding it alone (WavLM has no cross-
        sample interaction and this feature extractor does not per-sample normalize).
        Returns a list of [num_bins, L, D] (all) or [num_bins, D] (sum) float16 arrays,
        one per input, in order."""
        arr = [np.asarray(w, dtype=np.float32) for w in waveforms]
        inputs = self.feature_extractor(arr, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)  # [B, N]
        out = self.model(input_values)
        # hidden_states: tuple length L of [B, T, D] -> [B, T, L, D]
        hs = torch.stack(out.hidden_states, dim=0).permute(1, 2, 0, 3).contiguous()
        feats_b = hs.float().cpu().numpy()  # float32 intermediate [B, T, L, D]
        return [self._grid_from_frames(feats_b[b], num_bins, bin_dur) for b in range(feats_b.shape[0])]

    def encode(
        self,
        waveform: np.ndarray,
        num_bins: int = 120,
        bin_dur: float = 0.05,
    ) -> np.ndarray:
        """waveform: 1-D float32 @ sample_rate (exactly the 6 s clip).
        Returns [num_bins, L, D] (all) or [num_bins, D] (sum), float16.

        Thin wrapper over :meth:`encode_batch` (batch of 1) so the single- and
        batched-caching paths share one numerical implementation."""
        return self.encode_batch([waveform], num_bins=num_bins, bin_dur=bin_dur)[0]

    @property
    def num_layers(self) -> int:
        # base WavLM: 12 transformer layers + 1 embedding output = 13
        return self.model.config.num_hidden_layers + 1

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size


def load_clip(path, context_seconds: float = 6.0, sample_rate: int = 16000) -> np.ndarray:
    """Load a PRE-CUT human-only mix clip (already windowed, robot excluded).

    The upstream pipeline produced these; we do not re-mix headsets. Resamples if
    needed and fixes length to exactly context_seconds * sample_rate (pad/trim)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    n = int(round(context_seconds * sample_rate))
    if len(data) < n:
        data = np.pad(data, (0, n - len(data)))
    else:
        data = data[:n]
    return data.astype(np.float32)


def build_group_audio(
    corpus,
    meeting_id: str,
    human_letters: List[str],
    context_start: float,
    context_seconds: float = 6.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """DEPRECATED at runtime (audio now comes from pre-cut human_mix_clip). Retained
    only for fixture generation (tests/build_fixture.py)."""
    """
    Mean-mix the human speakers' headset WAVs over the window into one 16 kHz
    mono clip of exactly context_seconds (zero-padded if a file is short).
    Resamples if a file is not already at sample_rate.
    """
    import soundfile as sf

    n_samples = int(round(context_seconds * sample_rate))
    mix = np.zeros(n_samples, dtype=np.float32)
    n_used = 0
    for letter in human_letters:
        path = corpus.headset_wav(meeting_id, letter)
        if not path.exists():
            continue
        info = sf.info(str(path))
        sr = info.samplerate
        start = int(round(context_start * sr))
        stop = start + int(round(context_seconds * sr))
        start = max(0, start)
        stop = min(stop, info.frames)
        if stop <= start:
            continue
        data, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        # fix length
        if len(data) < n_samples:
            data = np.pad(data, (0, n_samples - len(data)))
        else:
            data = data[:n_samples]
        mix += data
        n_used += 1
    if n_used > 0:
        mix /= n_used
    return mix
