#!/usr/bin/env python3
"""
Repack the per-file frozen-feature cache into ONE contiguous memmap per modality.

Why
---
Training is I/O-bound: ~808k tiny `audio/<sid>.npy` + `text/<sid>.npy` opens (small random
reads) dominate wall-clock (all cores busy but mostly idle, GPU ~0%, the net is ~1.6M params).
Packing each modality into a single [N, ...] memmap turns those random opens into sequential,
page-cacheable reads: the OS keeps hot rows in RAM, so warm epochs read from memory instead of
disk. This is a PURE I/O reorganization -- it only moves bytes, never re-encodes -- so the array
handed to the model is bit-identical to the per-file cache (proved by --verify: 200 random
sample_ids compared with np.array_equal). No training-logic or equivalence risk.

What it writes under --output (default: <cache-dir>/../cache_packed)
    audio.dat     float16 memmap, shape [N, *audio_per_sample_shape]  (dtype = source files')
    text.dat      float16 memmap, shape [N, *text_per_sample_shape]
    index.parquet sample_id -> row (shared across modalities) + split
    meta.json     dtype/shape per modality, N, row order source, status (resume marker)

Row order is the order of features_index.parquet (fixed/stable). A sample whose per-file feature
is missing gets a ZERO row for that modality -- exactly what the current loader returns for a
missing file -- so audio.dat / text.dat stay row-aligned. Idempotent & resumable: progress is
checkpointed in meta.json (rows_done); re-running finishes an interrupted pack. Chunked to bound
RAM.

Usage
    python scripts/pack_cache_memmap.py --cache-dir data/processed/cache        # pack + verify
    python scripts/pack_cache_memmap.py --cache-dir data/processed/cache --verify-only
    python scripts/pack_cache_memmap.py --cache-dir data/processed/cache --force  # re-pack
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.logging_setup import pbar, setup_logging

logger = logging.getLogger(__name__)

MODALITIES = ("audio", "text")


def _probe_modality(cache_dir: Path, index: pd.DataFrame, mod: str):
    """Return (per_sample_shape, dtype) from the first present file of `mod`, or (None, None)."""
    col = f"{mod}_path"
    for rel in index[col]:
        if rel is None or (isinstance(rel, float) and np.isnan(rel)):
            continue
        p = cache_dir / rel
        if p.exists():
            a = np.load(p)
            return tuple(a.shape), a.dtype
    return None, None


def _load_meta(packed_dir: Path):
    mp = packed_dir / "meta.json"
    return json.loads(mp.read_text()) if mp.exists() else None


def _write_meta(packed_dir: Path, meta: dict):
    (packed_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def pack(cache_dir: Path, packed_dir: Path, chunk: int = 2048, force: bool = False):
    index_path = cache_dir / "features_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found (run Phase 1 caching first).")
    index = pd.read_parquet(index_path).reset_index(drop=True)
    n = len(index)
    packed_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(packed_dir)
    if meta and meta.get("status") == "complete" and not force:
        logger.info("packed cache already complete (%d rows) at %s -- skipping pack.", meta["n"], packed_dir)
        return meta

    # discover per-sample shape + dtype per modality from the source files
    present = {}
    for mod in MODALITIES:
        shp, dt = _probe_modality(cache_dir, index, mod)
        if shp is None:
            logger.warning("modality '%s': no present source files -> not packing it "
                           "(loader will return zeros for it).", mod)
        else:
            present[mod] = {"shape": [n, *shp], "per_sample_shape": list(shp), "dtype": str(dt),
                            "path": f"{mod}.dat"}
            logger.info("modality '%s': %d rows x %s (%s) -> %s (%.1f GB)", mod, n, shp, dt,
                        f"{mod}.dat", n * int(np.prod(shp)) * np.dtype(dt).itemsize / 1e9)

    # write index up-front (order is fixed); row = position in features_index
    idx_out = pd.DataFrame({"sample_id": index["sample_id"].to_numpy(),
                            "split": index["split"].to_numpy() if "split" in index else "",
                            "row": np.arange(n, dtype=np.int64)})
    idx_out.to_parquet(packed_dir / "index.parquet", index=False)

    # resume marker
    fresh = not (meta and meta.get("status") == "in_progress" and not force
                 and meta.get("modalities", {}).keys() == present.keys())
    rows_done = 0 if (fresh or force) else int(meta.get("rows_done", 0))
    meta = {"status": "in_progress", "n": n, "modalities": present,
            "source_cache": str(cache_dir), "row_order": "features_index.parquet",
            "rows_done": rows_done}
    _write_meta(packed_dir, meta)

    # open memmaps (w+ for a fresh file, r+ to resume)
    mmaps, cols = {}, {}
    for mod, info in present.items():
        mode = "r+" if (not fresh and (packed_dir / info["path"]).exists()) else "w+"
        mmaps[mod] = np.memmap(packed_dir / info["path"], dtype=np.dtype(info["dtype"]),
                               mode=mode, shape=tuple(info["shape"]))
        cols[mod] = index[f"{mod}_path"].to_numpy()

    logger.info("packing rows [%d, %d) in chunks of %d ...", rows_done, n, chunk)
    for start in pbar(range(rows_done, n, chunk), desc="pack"):
        end = min(start + chunk, n)
        for mod, info in present.items():
            per = tuple(info["per_sample_shape"]); dt = np.dtype(info["dtype"])
            buf = np.zeros((end - start, *per), dtype=dt)
            for j in range(start, end):
                rel = cols[mod][j]
                if rel is None or (isinstance(rel, float) and np.isnan(rel)):
                    continue  # missing -> zero row (matches current loader)
                p = cache_dir / rel
                if p.exists():
                    a = np.load(p)
                    if a.shape != per:
                        raise ValueError(f"{p} shape {a.shape} != expected {per} for '{mod}'")
                    buf[j - start] = a
            mmaps[mod][start:end] = buf
        for mm in mmaps.values():
            mm.flush()
        meta["rows_done"] = end
        _write_meta(packed_dir, meta)

    for mm in mmaps.values():
        mm.flush(); del mm
    meta["status"] = "complete"
    _write_meta(packed_dir, meta)
    logger.info("pack complete: %d rows -> %s", n, packed_dir)
    return meta


def verify(cache_dir: Path, packed_dir: Path, n_samples: int = 200, seed: int = 0) -> bool:
    """Load features BOTH ways for n_samples random rows and assert bit-identical.

    This is the safety gate: if ANY sample differs, the packed cache is NOT safe to use.
    Compares raw on-disk dtype (float16) arrays with np.array_equal -- no tolerance."""
    meta = _load_meta(packed_dir)
    if not meta or meta.get("status") != "complete":
        logger.error("packed cache not complete; cannot verify."); return False
    index = pd.read_parquet(cache_dir / "features_index.parquet").reset_index(drop=True)
    packed_idx = pd.read_parquet(packed_dir / "index.parquet")
    row_of = dict(zip(packed_idx["sample_id"], packed_idx["row"]))

    mmaps = {mod: np.memmap(packed_dir / info["path"], dtype=np.dtype(info["dtype"]),
                            mode="r", shape=tuple(info["shape"]))
             for mod, info in meta["modalities"].items()}

    rng = np.random.default_rng(seed)
    n = len(index)
    pick = rng.choice(n, size=min(n_samples, n), replace=False)
    checked, mism = 0, 0
    for i in pbar(pick, desc="verify"):
        sid = index.iloc[int(i)]["sample_id"]
        for mod, info in meta["modalities"].items():
            rel = index.iloc[int(i)][f"{mod}_path"]
            per = tuple(info["per_sample_shape"]); dt = np.dtype(info["dtype"])
            if rel is None or (isinstance(rel, float) and np.isnan(rel)):
                ref = np.zeros(per, dtype=dt)                     # missing -> zeros both ways
            else:
                p = cache_dir / rel
                ref = np.load(p) if p.exists() else np.zeros(per, dtype=dt)
            got = np.asarray(mmaps[mod][row_of[sid]])
            checked += 1
            if not np.array_equal(ref, got):
                mism += 1
                logger.error("MISMATCH sid=%s mod=%s row=%d (ref%s got%s)",
                             sid, mod, row_of[sid], ref.shape, got.shape)
    if mism:
        logger.error("VERIFY FAILED: %d/%d comparisons differ -- DO NOT use the packed cache.",
                     mism, checked)
        return False
    logger.info("VERIFY OK: %d comparisons across %d samples are bit-identical (np.array_equal).",
                checked, len(pick))
    return True


def main(args):
    setup_logging(None, level=args.log_level)
    cache_dir = Path(args.cache_dir)
    packed_dir = Path(args.output) if args.output else cache_dir.parent / "cache_packed"

    if not args.verify_only:
        pack(cache_dir, packed_dir, chunk=args.chunk, force=args.force)
    ok = verify(cache_dir, packed_dir, n_samples=args.verify_samples, seed=args.seed)
    if not ok:
        sys.exit(1)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True, help="Per-file cache (audio/, text/, features_index.parquet).")
    p.add_argument("--output", default=None, help="Packed output dir (default <cache-dir>/../cache_packed).")
    p.add_argument("--chunk", type=int, default=2048, help="Rows per write chunk (bounds RAM).")
    p.add_argument("--force", action="store_true", help="Re-pack from scratch even if complete.")
    p.add_argument("--verify-only", action="store_true", help="Skip packing; only run the equivalence check.")
    p.add_argument("--verify-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
