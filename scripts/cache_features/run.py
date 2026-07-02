#!/usr/bin/env python3
"""
Phase 1: cache frozen-encoder features (WavLM audio + RoBERTa text) per sample.

Audio  = the PRE-CUT human-only mix clip `human_mix_clip` from the manifest (already
         windowed to [context_start, context_end], pseudo-robot excluded). We do NOT
         re-mix headsets. Samples with missing_audio=true are skipped (logged).
Text   = the manifest `text_context` events (human speakers, end <= t). AMI words XML
         is only a fallback behind --text-source ami_words.

The manifest is read via the single loader with the future_for_labeling_only tripwire
ON, so any accidental read of label-only fields raises. Optionally cross-checks text
coverage against speech_regions and warns if a speaker active in the window is missing
from text_context.

On-disk precision: feature arrays are written as **float16** (frozen features feed a
GRU; float16 halves the cache -- ~285 GB instead of ~570 GB for 808k samples -- and the
loader up-casts to float32). All encoder math (WavLM/RoBERTa forward, pooling, layer
weighting) stays float32; only the array that hits disk is float16.

Throughput: per-sample CPU prep (clip load + resample + AMI word collection) runs in a
DataLoader with several workers so it overlaps the GPU, and encoder forwards are batched
(--batch-size) so the GPU stays saturated instead of sawtoothing on serial single-sample
work. Output is identical (modulo float16) and independent of batch/worker settings --
each sid always writes audio/<sid>.npy and text/<sid>.npy.

Resumable: a sample whose expected output file(s) already exist is skipped and its index
row is reconstructed, so an interrupted run can be re-invoked to finish (use --overwrite
to force re-encoding).

Writes under --output: audio/<sid>.npy, text/<sid>.npy, features_index.parquet,
cache_meta.json.

Usage:
    python -m scripts.cache_features.run --manifest <jsonl> --output data/processed/cache \
        --modality both [--clips-root .] [--speech-regions <pq>] [--text-source manifest_events|ami_words] \
        [--device cuda] [--batch-size 48] [--num-workers 10]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.audio_wavlm import WavLMCacher, load_clip
from src.features.text_roberta import RoBERTaCacher, collect_human_words, events_to_words
from src.utils.logging_setup import pbar, setup_logging
from src.utils.manifest import load_records

logger = logging.getLogger(__name__)

# On-disk dtype for every cached feature array. Single source of truth so the log
# message, the cache_meta.json record, and the actual np.save all agree.
FEATURE_DTYPE = np.float16


def _worker_init(worker_id):
    """Cap per-worker threads so N data workers don't oversubscribe the CPU while the
    GPU is the bottleneck (12 workers each spawning BLAS/OMP thread pools would thrash)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # pragma: no cover
        pass


def _identity_collate(batch):
    """Keep the list of prepared per-sample dicts as-is; the main process does the
    (batched) GPU encode and the saving. A real function (not a lambda) so it survives
    a 'spawn' start method too."""
    return batch


class SamplePrep:
    """Torch Dataset that does ONLY the CPU-side per-sample prep -- clip load/resample
    and causal word collection -- so DataLoader workers overlap it with the GPU. No
    encoding here; each __getitem__ returns picklable primitives + numpy/TimedWord.

    On Linux (fork start method) the records list, the AMICorpus (with its per-meeting
    word-parse memoization), and the lazily-built SpeechRegions are inherited by workers
    -- no re-pickling of the dataset, and word XML is parsed at most once per worker."""

    def __init__(self, records, *, do_audio, do_text, text_source, clips_root,
                 context_seconds, sample_rate, corpus, regions_path):
        self.records = records
        self.do_audio = do_audio
        self.do_text = do_text
        self.text_source = text_source
        self.clips_root = Path(clips_root)
        self.context_seconds = context_seconds
        self.sample_rate = sample_rate
        self.corpus = corpus
        self.regions_path = regions_path
        self._regions = None  # lazily built per worker process (see _get_regions)

    def __len__(self):
        return len(self.records)

    def _get_regions(self):
        if self.regions_path is None:
            return None
        if self._regions is None:
            from src.data.speech_regions import SpeechRegions
            self._regions = SpeechRegions(self.regions_path)
        return self._regions

    def __getitem__(self, i):
        rec = self.records[i]
        item = {
            "sid": rec.sample_id, "split": rec.split,
            "wav": None, "audio_produced": False,
            "words": None, "context_start": rec.context_start,
            "coverage_missing": None, "error": None,
        }
        try:
            if self.do_audio:
                if rec.missing_audio or not rec.human_mix_clip:
                    item["audio_produced"] = False  # missing_audio -> no audio file
                else:
                    clip_path = self.clips_root / rec.human_mix_clip
                    item["wav"] = load_clip(clip_path, context_seconds=self.context_seconds,
                                            sample_rate=self.sample_rate)
                    item["audio_produced"] = True

            if self.do_text:
                if self.text_source == "manifest_events":
                    events = rec.human_text_events(rec.time)
                    regions = self._get_regions()
                    if regions is not None:
                        ev_spk = {e["speaker"] for e in events}
                        active = {sp for sp in rec.human_speakers
                                  if regions.regions_before(rec.meeting_id, sp, rec.time)
                                  and any(s < rec.time and e > rec.context_start
                                          for s, e in regions.regions_before(rec.meeting_id, sp, rec.time))}
                        missing = active - ev_spk
                        if missing:
                            item["coverage_missing"] = sorted(missing)
                    item["words"] = events_to_words(events)
                else:  # ami_words fallback
                    item["words"] = collect_human_words(
                        self.corpus, rec.meeting_id, rec.human_speakers,
                        rec.context_start, rec.time)
        except Exception as e:  # noqa: BLE001 -- prep error isolated to this sample
            item["error"] = repr(e)
        return item


def _expected_rel(rec, do_audio, do_text):
    """The relative output paths this record SHOULD produce (None where it produces
    nothing -- missing_audio, or a disabled modality)."""
    a = f"audio/{rec.sample_id}.npy" if (do_audio and not rec.missing_audio and rec.human_mix_clip) else None
    t = f"text/{rec.sample_id}.npy" if do_text else None
    return a, t


def main(args):
    out_dir = Path(args.output)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, level=args.log_level, filename="cache.log")

    records = load_records(args.manifest, tripwire_future=True)  # leakage tripwire ON
    if args.max_samples is not None:
        records = records[: args.max_samples]

    do_audio = args.modality in ("audio", "both")
    do_text = args.modality in ("text", "both")

    logger.info("cache | %d samples | modality=%s | text_source=%s | on-disk dtype=%s (float16) | "
                "batch_size=%d | num_workers=%d",
                len(records), args.modality, args.text_source,
                np.dtype(FEATURE_DTYPE).name, args.batch_size, args.num_workers)

    # --- resumability: skip records whose expected output(s) already exist ----------
    index_rows = []
    todo = []
    resumed = 0
    for rec in records:
        a_rel, t_rel = _expected_rel(rec, do_audio, do_text)
        done = ((a_rel is None or (out_dir / a_rel).exists())
                and (t_rel is None or (out_dir / t_rel).exists()))
        if done and not args.overwrite:
            resumed += 1
            index_rows.append({"sample_id": rec.sample_id, "split": rec.split,
                               "audio_path": a_rel if a_rel else None,
                               "text_path": t_rel if t_rel else None})
        else:
            todo.append(rec)
    if resumed:
        logger.info("resume | %d already cached (skipped) | %d to process", resumed, len(todo))

    # --- encoders (frozen, GPU) -----------------------------------------------------
    layer_weights = None
    if args.layer_weights:
        layer_weights = [float(x) for x in args.layer_weights.split(",") if x.strip() != ""]
    audio_cacher = (WavLMCacher(model_name=args.wavlm_model, device=args.device,
                                layer_mode=args.layer_mode, layer_weights=layer_weights)
                    if do_audio else None)
    text_cacher = RoBERTaCacher(model_name=args.roberta_model, device=args.device) if do_text else None

    corpus = None
    if do_text and args.text_source == "ami_words":
        from src.data.ami import AMICorpus
        corpus = AMICorpus(args.ami_root)
    regions_path = args.speech_regions if (do_text and args.text_source == "manifest_events") else None

    errors, skipped_audio, coverage_warns, processed = [], 0, 0, 0

    if todo:
        prep = SamplePrep(todo, do_audio=do_audio, do_text=do_text, text_source=args.text_source,
                          clips_root=args.clips_root, context_seconds=args.context_seconds,
                          sample_rate=args.sample_rate, corpus=corpus, regions_path=regions_path)
        loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         collate_fn=_identity_collate)
        if args.num_workers > 0:
            loader_kw.update(prefetch_factor=args.prefetch_factor, persistent_workers=True,
                             pin_memory=(args.device != "cpu"), worker_init_fn=_worker_init)
        from torch.utils.data import DataLoader
        loader = DataLoader(prep, **loader_kw)

        t0 = time.time()
        for batch in pbar(loader, total=len(loader), desc=f"cache:{args.modality}"):
            # audio: one WavLM forward for all produced clips in the batch
            if do_audio:
                a_idx = [j for j, it in enumerate(batch) if it["error"] is None and it["audio_produced"]]
                if a_idx:
                    try:
                        wavs = [batch[j]["wav"] for j in a_idx]
                        grids = audio_cacher.encode_batch(wavs, num_bins=args.num_frames,
                                                          bin_dur=args.frame_shift)
                        for k, j in enumerate(a_idx):
                            np.save(out_dir / f"audio/{batch[j]['sid']}.npy",
                                    grids[k].astype(FEATURE_DTYPE))
                    except Exception as e:  # noqa: BLE001
                        for j in a_idx:
                            errors.append((batch[j]["sid"], f"audio_encode: {e!r}"))
                            batch[j]["error"] = repr(e)
                        logger.error("audio batch encode failed (%d samples): %s", len(a_idx), e)

            # text: one RoBERTa forward for all non-errored samples in the batch
            if do_text:
                t_idx = [j for j, it in enumerate(batch) if it["error"] is None]
                if t_idx:
                    try:
                        tw_list = [batch[j]["words"] for j in t_idx]
                        cstarts = [batch[j]["context_start"] for j in t_idx]
                        grids = text_cacher.encode_sample_batch(tw_list, cstarts,
                                                                num_bins=args.num_frames,
                                                                bin_dur=args.frame_shift)
                        for k, j in enumerate(t_idx):
                            np.save(out_dir / f"text/{batch[j]['sid']}.npy",
                                    grids[k].astype(FEATURE_DTYPE))
                    except Exception as e:  # noqa: BLE001
                        for j in t_idx:
                            errors.append((batch[j]["sid"], f"text_encode: {e!r}"))
                            batch[j]["error"] = repr(e)
                        logger.error("text batch encode failed (%d samples): %s", len(t_idx), e)

            # bookkeeping + index rows (per sample, keyed by sid -- batch-order-independent)
            for it in batch:
                sid = it["sid"]
                if it["error"] is not None:
                    if not any(s == sid for s, _ in errors):
                        errors.append((sid, it["error"]))
                        logger.error("%s: %s", sid, it["error"])
                    continue
                if do_audio and not it["audio_produced"]:
                    skipped_audio += 1
                    logger.info("skip audio (missing_audio) for %s", sid)
                if it["coverage_missing"]:
                    coverage_warns += 1
                    logger.warning("%s: text_context missing speakers active in window: %s "
                                   "(consider --text-source ami_words)", sid, it["coverage_missing"])
                index_rows.append({
                    "sample_id": sid, "split": it["split"],
                    "audio_path": (f"audio/{sid}.npy" if (do_audio and it["audio_produced"]) else None),
                    "text_path": (f"text/{sid}.npy" if do_text else None),
                })
                processed += 1

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else float("nan")
        logger.info("processed %d samples in %.1fs -> %.1f samples/sec "
                    "(batch_size=%d, num_workers=%d)", processed, elapsed, rate,
                    args.batch_size, args.num_workers)

    pd.DataFrame(index_rows).to_parquet(out_dir / "features_index.parquet", index=False)
    meta = {
        "wavlm_model": args.wavlm_model if do_audio else None,
        "roberta_model": args.roberta_model if do_text else None,
        "layer_mode": args.layer_mode if do_audio else None,
        "num_layers": audio_cacher.num_layers if do_audio else None,
        "layer_weights": (audio_cacher.layer_weights.tolist()
                          if do_audio and audio_cacher.layer_weights is not None else None),
        "audio_dim": audio_cacher.hidden_dim if do_audio else None,
        "text_dim": text_cacher.hidden_dim if do_text else None,
        "text_source": args.text_source,
        "num_frames": args.num_frames, "frame_shift": args.frame_shift,
        "context_seconds": args.context_seconds, "sample_rate": args.sample_rate,
        "feature_dtype": np.dtype(FEATURE_DTYPE).name,  # on-disk precision (loader up-casts to float32)
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("indexed %d (%d newly processed, %d resumed) | audio skipped(missing): %d | "
                "coverage warns: %d | errors: %d -> %s",
                len(index_rows), processed, resumed, skipped_audio, coverage_warns, len(errors),
                out_dir / "features_index.parquet")
    if errors and args.strict:
        raise SystemExit(f"{len(errors)} sample(s) failed and --strict is set.")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="Cache dir.")
    p.add_argument("--modality", choices=["audio", "text", "both"], default="both")
    p.add_argument("--clips-root", default=".", help="Root to resolve human_mix_clip relative paths.")
    p.add_argument("--text-source", choices=["manifest_events", "ami_words"], default="manifest_events")
    p.add_argument("--speech-regions", default=None, help="For the text-coverage cross-check.")
    p.add_argument("--ami-root", default=None, help="Only needed for --text-source ami_words.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer-mode", choices=["all", "sum"], default="all")
    p.add_argument("--layer-weights", default=None,
                   help="Comma-separated fixed WavLM layer weights for --layer-mode sum "
                        "(length = num hidden states = 13; normalized in code). "
                        "Omit to use the default layer 3-8 mean prior. "
                        "Example: 0,0,0,1,1,1,1,1,1,0,0,0,0")
    p.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    p.add_argument("--roberta-model", default="roberta-base")
    p.add_argument("--num-frames", type=int, default=120)
    p.add_argument("--frame-shift", type=float, default=0.05)
    p.add_argument("--context-seconds", type=float, default=6.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    # throughput knobs: overlap CPU prep with the GPU and batch the encoder forwards
    p.add_argument("--batch-size", type=int, default=48, help="Encoder forward batch size (32-64 typical).")
    p.add_argument("--num-workers", type=int, default=10, help="DataLoader CPU-prep workers (8-12 typical).")
    p.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per worker.")
    p.add_argument("--overwrite", action="store_true", help="Re-encode even if the output file exists.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--strict", action="store_true")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
