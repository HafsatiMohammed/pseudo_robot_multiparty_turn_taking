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
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import TimingDataset
from .loaders import collate_batch

logger = logging.getLogger(__name__)


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
        packed_dir: Optional[str] = None,
        read_modalities: Optional[set] = None,
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
        # Modality gating: only the modalities in this set are actually read; others are
        # returned as zeros. Safe because GroupTurnFuse zeros every disabled modality at its
        # branch input anyway (see _maybe_zero), so a gated zero and a read-then-zeroed real
        # feature produce identical model output. None => read both (backward compatible).
        self.read_modalities = set(read_modalities) if read_modalities is not None else None

        meta_path = self.cache_dir / "cache_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.num_frames = meta.get("num_frames", frame_seq_len)
        self.layer_mode = meta.get("layer_mode") or "all"
        self.num_layers = meta.get("num_layers") or default_num_layers
        self.audio_dim = meta.get("audio_dim") or default_audio_dim
        self.text_dim = meta.get("text_dim") or default_text_dim

        if self.layer_mode == "all":
            self.audio_zero_shape = (self.num_frames, self.num_layers, self.audio_dim)
        else:
            self.audio_zero_shape = (self.num_frames, self.audio_dim)
        self.text_zero_shape = (self.num_frames, self.text_dim)

        # Cache is written as float16 (see scripts.cache_features.run); we up-cast to
        # float32 on load for the GRU. Log the on-disk dtype once per split so any
        # silent dtype drift (e.g. an old float32 cache) is visible in the logs.
        self._logged_disk_dtype = False
        disk_dtype = meta.get("feature_dtype")
        if disk_dtype is not None:
            logger.info("%s: cached features on disk as %s -> up-cast to float32 on load",
                        split, disk_dtype)

        # Packed (memmap) cache OR per-file cache. Packed turns ~808k tiny random .npy reads
        # into sequential, page-cacheable reads from one file per modality -- pure I/O change,
        # byte-identical to per-file (proved by scripts/pack_cache_memmap.py --verify).
        self.packed = False
        self._mmaps = {}          # opened lazily per worker process (fork-safe)
        if packed_dir is not None:
            self._init_packed(Path(packed_dir))
        if not self.packed:
            self._init_per_file()

    def _init_per_file(self) -> None:
        """Per-file cache: map sample_id -> relative audio/text .npy path."""
        import pandas as pd
        index_path = self.cache_dir / "features_index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(
                f"features_index.parquet not found in {self.cache_dir}. Run Phase 1 "
                "(scripts.cache_features.run) first."
            )
        index = pd.read_parquet(index_path)
        self.audio_paths = dict(zip(index["sample_id"], index["audio_path"]))
        self.text_paths = dict(zip(index["sample_id"], index["text_path"]))

    def _init_packed(self, packed_dir: Path) -> None:
        """Memmap cache: load meta + sample_id -> row index; shapes are authoritative."""
        import pandas as pd
        meta = json.loads((packed_dir / "meta.json").read_text())
        if meta.get("status") != "complete":
            raise RuntimeError(
                f"Packed cache at {packed_dir} is not complete (status={meta.get('status')}). "
                "Finish scripts/pack_cache_memmap.py before training against it."
            )
        self.packed = True
        self.packed_dir = packed_dir
        self.packed_meta = meta["modalities"]
        idx = pd.read_parquet(packed_dir / "index.parquet")
        self.packed_row = dict(zip(idx["sample_id"], idx["row"]))
        # Use the packed per-sample shapes for zeros so gated/missing rows match real rows.
        if "audio" in self.packed_meta:
            self.audio_zero_shape = tuple(self.packed_meta["audio"]["per_sample_shape"])
        if "text" in self.packed_meta:
            self.text_zero_shape = tuple(self.packed_meta["text"]["per_sample_shape"])
        logger.info("%s: using packed memmap cache at %s (modalities=%s)",
                    self.split, packed_dir, list(self.packed_meta))

    def _memmap(self, mod: str) -> np.memmap:
        """Open (once per process) the read-only memmap for a modality."""
        mm = self._mmaps.get(mod)
        if mm is None:
            info = self.packed_meta[mod]
            mm = np.memmap(self.packed_dir / info["path"], dtype=np.dtype(info["dtype"]),
                           mode="r", shape=tuple(info["shape"]))
            self._mmaps[mod] = mm
        return mm

    def _get_modality(self, sid, mod: str, zero_shape) -> np.ndarray:
        """Return this sample's modality feature as a float32 array (zeros if gated/missing).

        Bit-identical to the per-file path: packed rows hold the SAME on-disk bytes as the
        .npy files and are up-cast to float32 the same way."""
        # Gated modalities are never read (the model zeros them regardless).
        if self.read_modalities is not None and mod not in self.read_modalities:
            return np.zeros(zero_shape, dtype=np.float32)
        if self.packed:
            row = self.packed_row.get(sid)
            if row is None or mod not in self.packed_meta:
                return np.zeros(zero_shape, dtype=np.float32)
            # np.array (not asarray) => always a writable, contiguous copy, so torch.from_numpy
            # never shares memory with the read-only memmap and never warns. Same up-cast to
            # float32 as the per-file path's .astype, so the bytes fed to the model are identical.
            return np.array(self._memmap(mod)[int(row)], dtype=np.float32)
        rel = (self.audio_paths if mod == "audio" else self.text_paths).get(sid)
        return self._load_cached(rel, zero_shape)

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
        raw = np.load(p)  # float16 on disk (frozen features); up-cast for the GRU
        if not self._logged_disk_dtype:
            logger.info("%s: loaded cached feature %s with on-disk dtype=%s -> float32",
                        self.split, rel, raw.dtype)
            self._logged_disk_dtype = True
        return raw.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sid = item["sample_id"]
        audio = self._get_modality(sid, "audio", self.audio_zero_shape)
        text = self._get_modality(sid, "text", self.text_zero_shape)
        item["audio"] = torch.from_numpy(audio)
        item["text"] = torch.from_numpy(text)
        # Guard against silent dtype drift: the GRU expects float32, regardless of the
        # float16 on-disk cache.
        assert item["audio"].dtype == torch.float32 and item["text"].dtype == torch.float32, \
            f"frozen features must up-cast to float32 (got audio={item['audio'].dtype}, text={item['text'].dtype})"
        return item


def collate_multimodal(batch):
    """Stack all four modalities. Reuses collate_batch for timing/scalar/label/weight."""
    out = collate_batch(batch)
    out["audio"] = torch.stack([b["audio"] for b in batch])  # [B,T,L,D] or [B,T,D]
    out["text"] = torch.stack([b["text"] for b in batch])    # [B,T,768]
    return out


def resolve_packed_dir(cache_dir, cache_format: str = "auto", packed_dir: Optional[str] = None):
    """Decide whether to use a packed (memmap) cache.

    cache_format: 'auto' (use packed if a COMPLETE one exists, else per-file), 'memmap'
    (require it), 'per_file' (never). Default packed location is <cache_dir>/../cache_packed.
    """
    if cache_format == "per_file":
        return None
    cand = Path(packed_dir) if packed_dir else Path(cache_dir).parent / "cache_packed"
    meta = cand / "meta.json"
    complete = meta.exists() and json.loads(meta.read_text()).get("status") == "complete"
    if complete:
        return cand
    if cache_format == "memmap":
        raise FileNotFoundError(
            f"cache_format='memmap' but no complete packed cache at {cand}. "
            "Run scripts/pack_cache_memmap.py first."
        )
    return None  # auto: fall back to the per-file cache


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
    cache_format: str = "auto",
    packed_dir: Optional[str] = None,
    read_modalities: Optional[set] = None,
) -> Dict[str, object]:
    """
    Build train/val/test multimodal dataloaders. Val/test reuse the TRAIN
    normalization stats by default (avoids per-split normalization drift).

    cache_format/packed_dir select the packed (memmap) cache; read_modalities restricts
    which modalities are read from it (the rest are zeros -- the model zeros disabled
    modalities anyway). See resolve_packed_dir and MultimodalDataset.
    """
    if val_batch_size is None:
        val_batch_size = batch_size * 2
    if test_batch_size is None:
        test_batch_size = batch_size * 2

    torch.manual_seed(seed)
    np.random.seed(seed)

    resolved_packed = resolve_packed_dir(cache_dir, cache_format, packed_dir)
    if resolved_packed is not None:
        logger.info("multimodal cache: PACKED memmap at %s (read_modalities=%s)",
                    resolved_packed, "all" if read_modalities is None else sorted(read_modalities))
    ds_kw = dict(packed_dir=str(resolved_packed) if resolved_packed else None,
                 read_modalities=read_modalities)

    timing_dir = Path(timing_dir)
    train_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "train.parquet"), cache_dir=cache_dir,
        split="train", normalize=normalize, require_cache=require_cache, **ds_kw,
    ).truncate(max_samples)
    stats = train_ds.norm_stats() if (normalize and reuse_train_norm) else None
    val_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "validation.parquet"), cache_dir=cache_dir,
        split="validation", normalize=normalize, require_cache=require_cache, norm_stats=stats, **ds_kw,
    ).truncate(max_samples)
    test_ds = MultimodalDataset(
        parquet_path=str(timing_dir / "test.parquet"), cache_dir=cache_dir,
        split="test", normalize=normalize, require_cache=require_cache, norm_stats=stats, **ds_kw,
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
