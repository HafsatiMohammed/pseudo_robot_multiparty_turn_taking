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
  layer_mode="sum" -> [120, D]    (sum across layers; smaller on disk)
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .align import pool_frames_to_grid, wavlm_frame_times


class WavLMCacher:
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: str = "cpu",
        layer_mode: str = "all",
        sample_rate: int = 16000,
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

    @torch.no_grad()
    def encode(
        self,
        waveform: np.ndarray,
        num_bins: int = 120,
        bin_dur: float = 0.05,
    ) -> np.ndarray:
        """waveform: 1-D float32 @ sample_rate (exactly the 6 s clip).
        Returns [num_bins, L, D] (all) or [num_bins, D] (sum)."""
        inputs = self.feature_extractor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        out = self.model(input_values)
        # hidden_states: tuple length L of [1, T, D]
        hs = torch.stack(out.hidden_states, dim=0)  # [L, 1, T, D]
        hs = hs.squeeze(1).permute(1, 0, 2).contiguous()  # [T, L, D]
        feats = hs.cpu().numpy().astype(np.float32)

        T = feats.shape[0]
        ftimes = wavlm_frame_times(T)
        grid = pool_frames_to_grid(feats, ftimes, num_bins=num_bins, bin_dur=bin_dur)  # [120,L,D]
        if self.layer_mode == "sum":
            grid = grid.sum(axis=1).astype(np.float32)  # [120, D]
        return grid

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
