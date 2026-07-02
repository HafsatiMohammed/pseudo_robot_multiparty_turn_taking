"""
Phase 2: multimodal data layer.

MultimodalDataset extends TimingDataset: it keeps the timing X_frame [120,7] and
X_scalar [6] (normalized exactly as before) and additionally loads the cached,
FROZEN audio [120,L,D]|[120,D] and text [120,768] embeddings keyed by sample_id
via features_index.parquet.

Contract:
  - Frozen embeddings are NOT normalized (they are fixed encoder outputs).
  - Every item always returns all four modalities (real tensors or zeros) so the
    model -- not the dataloader -- enforces the ablation (matched capacity).
  - The weighted sampler (class balancing) is for the TRAIN split only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import TimingDataset
from .loaders import collate_batch


class MultimodalDataset(TimingDataset):
    def __init__(
        self,
        parquet_path: str,
        cache_dir: str,
        split: str = "train",
        frame_seq_len: int = 120,
        frame_dim: int = 7,
        scalar_dim: int = 6,
        normalize: bool = True,
        require_cache: bool = True,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        default_audio_dim: int = 768,
        default_text_dim: int = 768,
        default_num_layers: int = 13,
    ):
        super().__init__(
            parquet_path=parquet_path,
            split=split,
            frame_seq_len=frame_seq_len,
            frame_dim=frame_dim,
            scalar_dim=scalar_dim,
            normalize=normalize,
        )
        # Optionally override per-split normalization with fixed (train) stats.
        if normalize and norm_stats is not None:
            self.frame_mean = norm_stats["frame_mean"]
            self.frame_std = norm_stats["frame_std"]
            self.scalar_mean = norm_stats["scalar_mean"]
            self.scalar_std = norm_stats["scalar_std"]

        self.cache_dir = Path(cache_dir)
        self.require_cache = require_cache

        meta_path = self.cache_dir / "cache_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.num_frames = meta.get("num_frames", frame_seq_len)
        self.layer_mode = meta.get("layer_mode") or "all"
        self.num_layers = meta.get("num_layers") or default_num_layers
        self.audio_dim = meta.get("audio_dim") or default_audio_dim
        self.text_dim = meta.get("text_dim") or default_text_dim

        index_path = self.cache_dir / "features_index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(
                f"features_index.parquet not found in {self.cache_dir}. Run Phase 1 "
                "(scripts.cache_features.run) first."
            )
        import pandas as pd

        index = pd.read_parquet(index_path)
        self.audio_paths = dict(zip(index["sample_id"], index["audio_path"]))
        self.text_paths = dict(zip(index["sample_id"], index["text_path"]))

        if self.layer_mode == "all":
            self.audio_zero_shape = (self.num_frames, self.num_layers, self.audio_dim)
        else:
            self.audio_zero_shape = (self.num_frames, self.audio_dim)
        self.text_zero_shape = (self.num_frames, self.text_dim)

    def truncate(self, n: Optional[int]) -> "MultimodalDataset":
        """Keep only the first n rows (fast dry runs). Labels/weights stay aligned."""
        if n is not None and n < len(self.df):
            self.df = self.df.iloc[:n].reset_index(drop=True)
            self.labels = self.labels[:n]
            self.weights = self.weights[:n]
        return self

    def norm_stats(self) -> Dict[str, np.ndarray]:
        """Expose this split's normalization stats (so train stats can be reused)."""
        return {
            "frame_mean": self.frame_mean,
            "frame_std": self.frame_std,
            "scalar_mean": self.scalar_mean,
            "scalar_std": self.scalar_std,
        }

    def _load_cached(self, rel: Optional[str], zero_shape) -> np.ndarray:
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            return np.zeros(zero_shape, dtype=np.float32)
        p = self.cache_dir / rel
        if not p.exists():
            if self.require_cache:
                raise FileNotFoundError(f"Cached feature missing: {p}")
            return np.zeros(zero_shape, dtype=np.float32)
        arr = np.load(p).astype(np.float32)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sid = item["sample_id"]
        audio = self._load_cached(self.audio_paths.get(sid), self.audio_zero_shape)
        text = self._load_cached(self.text_paths.get(sid), self.text_zero_shape)
        item["audio"] = torch.from_numpy(audio)
        item["text"] = torch.from_numpy(text)
        return item


def collate_multimodal(batch):
    """Stack all four modalities. Reuses collate_batch for timing/scalar/label/weight."""
    out = collate_batch(batch)
    out["audio"] = torch.stack([b["audio"] for b in batch])  # [B,T,L,D] or [B,T,D]
    out["text"] = torch.stack([b["text"] for b in batch])    # [B,T,768]
    return out


def get_multimodal_dataloaders(
    timing_dir: str,
    cache_dir: str,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    normalize: bool = True,
    reuse_train_norm: bool = True,
    require_cache: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None,
    worker_init_fn=None,
    generator=None,
) -> Dict[str, object]:
    """
    Build train/val/test multimodal dataloaders. Val/test reuse the TRAIN
    normalization stats by default (avoids per-split normalization drift).
    """
    if val_batch_size is None:
        val_batch_size = batch_size * 2
    if test_batch_size is None:
        test_batch_size = batch_size * 2

    torch.manual_seed(seed)
    np.random.seed(seed)

    timing_dir = Path(timing_dir)
    train_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "train.parquet"), cache_dir=cache_dir,
        split="train", normalize=normalize, require_cache=require_cache,
    ).truncate(max_samples)
    stats = train_ds.norm_stats() if (normalize and reuse_train_norm) else None
    val_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "validation.parquet"), cache_dir=cache_dir,
        split="validation", normalize=normalize, require_cache=require_cache, norm_stats=stats,
    ).truncate(max_samples)
    test_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "test.parquet"), cache_dir=cache_dir,
        split="test", normalize=normalize, require_cache=require_cache, norm_stats=stats,
    ).truncate(max_samples)

    common = dict(num_workers=num_workers, collate_fn=collate_multimodal,
                  worker_init_fn=worker_init_fn)
    # drop a trailing size-1 batch (BatchNorm needs >1) -- but never drop the only
    # batch on a tiny dataset.
    drop_last = len(train_ds) > batch_size
    if use_weighted_sampler:
        cw = train_ds.get_class_weights()
        sample_w = cw[train_ds.labels]
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(train_ds), replacement=True,
                                        generator=generator)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  drop_last=drop_last, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  generator=generator, drop_last=drop_last, **common)

    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **common)

    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "train_dataset": train_ds, "val_dataset": val_ds, "test_dataset": test_ds,
    }
