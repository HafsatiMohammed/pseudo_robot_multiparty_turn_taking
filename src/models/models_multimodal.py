"""
Multimodal turn-taking models — GroupTurnFuse.

Design principle (matched capacity):
    All four ablation systems are the SAME network. A "system" only changes which
    modalities are active; disabled modalities are zeroed at their branch input, so
    parameter count and architecture are identical across rows and only information
    content varies. This removes the modality-vs-parameter-count confound.

Systems (use build_system):
    "timing"       : timing frames + scalars                 (timing-only)
    "audio_timing" : + WavLM audio                           (audio + timing)
    "text_timing"  : + text                                  (text + timing)
    "full"         : timing + scalars + audio + text         (GroupTurn-Fuse)

Reuses your existing branches: FrameBranchTCN, FrameBranchGRU, ScalarBranch,
FusionModule. Adds: WavLMLayerWeighting, AudioBranch, TextBranch, GatedFusion.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .branches import FrameBranchTCN, FrameBranchGRU, ScalarBranch, FusionModule


MODALITIES = ("timing", "scalar", "audio", "text")


def _make_temporal_encoder(cfg: Dict, in_channels: int, output_dim: int) -> nn.Module:
    """Build a TCN or GRU temporal encoder, reusing the frame-branch classes."""
    enc_type = cfg.get("type", "gru").lower()
    if enc_type == "tcn":
        return FrameBranchTCN(
            in_channels=in_channels,
            out_channels_list=cfg.get("out_channels", [64, 128]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.3),
            output_dim=output_dim,
        )
    if enc_type == "gru":
        return FrameBranchGRU(
            in_channels=in_channels,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            output_dim=output_dim,
            bidirectional=cfg.get("bidirectional", True),
        )
    raise ValueError(f"Unknown temporal encoder type: {enc_type}")


# ---------------------------------------------------------------------------
# WavLM layer weighting (learned weighted sum across hidden layers)
# ---------------------------------------------------------------------------
class WavLMLayerWeighting(nn.Module):
    """
    Learnable softmax-weighted sum across WavLM layers.
    Input  [B, T, L, D]  (all L hidden states, frozen, cached)
    Output [B, T, D]
    Use only if you cached all layers. If you cached a pre-summed [B, T, D]
    feature, skip this (AudioBranch detects the input rank automatically).
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)  # [L]
        return torch.einsum("btld,l->btd", x, w)


# ---------------------------------------------------------------------------
# Audio branch (WavLM frame features -> proj -> temporal encoder)
# ---------------------------------------------------------------------------
class AudioBranch(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.wavlm_dim = cfg.get("wavlm_dim", 768)          # 768 base / 1024 large
        self.proj_dim = cfg.get("proj_dim", 64)
        self.output_dim = cfg.get("output_dim", 128)
        self.use_layer_weighting = cfg.get("use_layer_weighting", False)

        if self.use_layer_weighting:
            self.layer_weighting = WavLMLayerWeighting(cfg.get("num_layers", 13))
        self.proj = nn.Linear(self.wavlm_dim, self.proj_dim)
        self.encoder = _make_temporal_encoder(
            cfg.get("encoder", {}), in_channels=self.proj_dim, output_dim=self.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, L, D] (all layers) or [B, T, D] (pre-summed)
        if x.dim() == 4:
            if not self.use_layer_weighting:
                raise ValueError("4D audio input but use_layer_weighting=False.")
            x = self.layer_weighting(x)
        x = self.proj(x)              # [B, T, proj]
        return self.encoder(x)        # [B, output_dim]


# ---------------------------------------------------------------------------
# Text branch (sequence of word/token embeddings, or a pooled context vector)
# ---------------------------------------------------------------------------
class TextBranch(nn.Module):
    """
    Two modes (set by config 'mode'):
      'sequence' : input [B, T, D] frame/word-aligned embeddings -> temporal
                   encoder. PREFERRED — preserves recency (the last clause carries
                   most of the turn-taking signal).
      'vector'   : input [B, D] a single pooled context embedding -> MLP.
                   Simpler but tends to wash out recency.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.mode = cfg.get("mode", "sequence").lower()
        self.text_dim = cfg.get("text_dim", 768)
        self.proj_dim = cfg.get("proj_dim", 64)
        self.output_dim = cfg.get("output_dim", 128)

        self.proj = nn.Linear(self.text_dim, self.proj_dim)
        if self.mode == "sequence":
            self.encoder = _make_temporal_encoder(
                cfg.get("encoder", {}), in_channels=self.proj_dim, output_dim=self.output_dim
            )
        elif self.mode == "vector":
            self.encoder = ScalarBranch(
                in_dim=self.proj_dim,
                hidden_dims=cfg.get("hidden_dims", [128, 128]),
                output_dim=self.output_dim,
                dropout=cfg.get("dropout", 0.3),
            )
        else:
            raise ValueError(f"Unknown text mode: {self.mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Gated fusion (per-sample modality gates -> inspection figure)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Projects each modality embedding to a common dim, computes a softmax gate over
    modalities, fuses by weighted sum, then classifies. Returns logits and the
    per-sample gates [B, M] (the gate-inspection signal: e.g. text up-weighted
    after questions, audio during overlap).
    """

    def __init__(self, modality_dims: Sequence[int], common_dim: int,
                 hidden_dims: Sequence[int], num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, common_dim) for d in modality_dims])
        self.gate = nn.Sequential(
            nn.Linear(common_dim * len(modality_dims), common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, len(modality_dims)),
        )
        self.classifier = FusionModule(
            in_dim=common_dim, hidden_dims=list(hidden_dims),
            output_dim=num_classes, dropout=dropout,
        )

    def forward(self, embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = [p(e) for p, e in zip(self.projs, embs)]   # each [B, common]
        gates = torch.softmax(self.gate(torch.cat(proj, dim=1)), dim=1)  # [B, M]
        fused = sum(gates[:, i : i + 1] * proj[i] for i in range(len(proj)))
        return self.classifier(fused), gates


# ---------------------------------------------------------------------------
# Unified model
# ---------------------------------------------------------------------------
class GroupTurnFuse(nn.Module):
    """
    One network, all branches. The active set of modalities is fixed per system;
    disabled modalities are zeroed at their branch input (matched capacity).
    """

    def __init__(self, config: Dict, active_modalities: Sequence[str] = MODALITIES):
        super().__init__()
        self.config = config
        self.active = set(active_modalities)
        bad = self.active - set(MODALITIES)
        if bad:
            raise ValueError(f"Unknown modalities: {bad}")
        if "timing" not in self.active or "scalar" not in self.active:
            # timing+scalar are the always-on base; warn but allow if you really want
            pass

        m = config.get("model", {})
        frame_cfg = m.get("frame_branch", {})
        scalar_cfg = m.get("scalar_branch", {})
        audio_cfg = m.get("audio_branch", {})
        text_cfg = m.get("text_branch", {})
        fusion_cfg = m.get("fusion", {})

        # branches (ALWAYS instantiated -> constant parameter count)
        self.frame_branch = _make_temporal_encoder(
            {**frame_cfg, "type": frame_cfg.get("type", "gru")},
            in_channels=frame_cfg.get("in_channels", 7),
            output_dim=frame_cfg.get("output_dim", 128),
        )
        self.scalar_branch = ScalarBranch(
            in_dim=scalar_cfg.get("in_dim", 6),
            hidden_dims=scalar_cfg.get("hidden_dims", [64, 64]),
            output_dim=scalar_cfg.get("output_dim", 64),
            dropout=scalar_cfg.get("dropout", 0.2),
        )
        self.audio_branch = AudioBranch(audio_cfg)
        self.text_branch = TextBranch(text_cfg)

        self.frame_dim = frame_cfg.get("output_dim", 128)
        self.scalar_dim = scalar_cfg.get("output_dim", 64)
        self.audio_dim = audio_cfg.get("output_dim", 128)
        self.text_dim = text_cfg.get("output_dim", 128)

        self.fusion_type = fusion_cfg.get("type", "concat").lower()
        dims = [self.frame_dim, self.scalar_dim, self.audio_dim, self.text_dim]
        if self.fusion_type == "gated":
            self.fusion = GatedFusion(
                modality_dims=dims,
                common_dim=fusion_cfg.get("common_dim", 128),
                hidden_dims=fusion_cfg.get("hidden_dims", [128]),
                num_classes=3,
                dropout=fusion_cfg.get("dropout", 0.3),
            )
        elif self.fusion_type == "concat":
            self.fusion = FusionModule(
                in_dim=sum(dims),
                hidden_dims=fusion_cfg.get("hidden_dims", [256, 128]),
                output_dim=3,
                dropout=fusion_cfg.get("dropout", 0.3),
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _maybe_zero(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Zero a modality input if it is not active for this system."""
        return x if name in self.active else torch.zeros_like(x)

    def forward(
        self,
        frame: torch.Tensor,
        scalar: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        return_gates: bool = False,
    ):
        """
        All four inputs must be provided (real tensors, correct shapes). Inactive
        modalities are zeroed internally, so passing real features for a disabled
        modality cannot leak — the ablation is enforced here, not in the dataloader.
        """
        f = self.frame_branch(self._maybe_zero(frame, "timing"))
        s = self.scalar_branch(self._maybe_zero(scalar, "scalar"))
        a = self.audio_branch(self._maybe_zero(audio, "audio"))
        t = self.text_branch(self._maybe_zero(text, "text"))

        if self.fusion_type == "gated":
            logits, gates = self.fusion([f, s, a, t])
            return (logits, gates) if return_gates else logits

        fused = torch.cat([f, s, a, t], dim=1)
        return self.fusion(fused)

    @torch.no_grad()
    def get_probabilities(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> torch.Tensor:
        out = self.forward(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.argmax(logits, dim=1)


# ---------------------------------------------------------------------------
# System builders — the four ablation rows, all the same network
# ---------------------------------------------------------------------------
SYSTEM_MODALITIES = {
    "timing": ("timing", "scalar"),
    "audio_timing": ("timing", "scalar", "audio"),
    "text_timing": ("timing", "scalar", "text"),
    "full": ("timing", "scalar", "audio", "text"),
}


def build_system(name: str, config: Dict) -> GroupTurnFuse:
    """Build one ablation system. All share identical architecture/parameters."""
    if name not in SYSTEM_MODALITIES:
        raise ValueError(f"Unknown system '{name}'. Options: {list(SYSTEM_MODALITIES)}")
    return GroupTurnFuse(config, active_modalities=SYSTEM_MODALITIES[name])


def build_all_systems(config: Dict) -> Dict[str, GroupTurnFuse]:
    """Build all four systems for the ablation table."""
    return {name: build_system(name, config) for name in SYSTEM_MODALITIES}


# ---------------------------------------------------------------------------
# Example config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "model": {
        "frame_branch": {"type": "gru", "in_channels": 7, "hidden_dim": 128,
                         "num_layers": 2, "dropout": 0.3, "output_dim": 128,
                         "bidirectional": True},
        "scalar_branch": {"in_dim": 6, "hidden_dims": [64, 64], "output_dim": 64,
                          "dropout": 0.2},
        # layer_mode=sum caches a collapsed [T,768] audio rep -> no learnable layer
        # weighting. Flip to use_layer_weighting=True, num_layers=13 for layer_mode=all.
        "audio_branch": {"wavlm_dim": 768, "proj_dim": 64, "output_dim": 128,
                         "use_layer_weighting": False, "num_layers": 1,
                         "encoder": {"type": "gru", "hidden_dim": 128, "num_layers": 2,
                                     "dropout": 0.3, "bidirectional": True}},
        "text_branch": {"mode": "sequence", "text_dim": 768, "proj_dim": 64,
                        "output_dim": 128,
                        "encoder": {"type": "gru", "hidden_dim": 128, "num_layers": 2,
                                    "dropout": 0.3, "bidirectional": True}},
        "fusion": {"type": "gated", "common_dim": 128, "hidden_dims": [128],
                   "dropout": 0.3},
    }
}
