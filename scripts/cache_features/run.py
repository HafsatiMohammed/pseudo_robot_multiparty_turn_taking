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

Writes under --output: audio/<sid>.npy, text/<sid>.npy, features_index.parquet,
cache_meta.json.

Usage:
    python -m scripts.cache_features.run --manifest <jsonl> --output data/processed/cache \
        --modality both [--clips-root .] [--speech-regions <pq>] [--text-source manifest_events|ami_words]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.audio_wavlm import WavLMCacher, load_clip
from src.features.text_roberta import RoBERTaCacher, events_to_words
from src.utils.logging_setup import pbar, setup_logging
from src.utils.manifest import load_records

logger = logging.getLogger(__name__)


def main(args):
    out_dir = Path(args.output)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, level=args.log_level, filename="cache.log")

    records = load_records(args.manifest, tripwire_future=True)  # leakage tripwire ON
    if args.max_samples is not None:
        records = records[: args.max_samples]
    logger.info("cache | %d samples | modality=%s | text_source=%s", len(records), args.modality, args.text_source)

    do_audio = args.modality in ("audio", "both")
    do_text = args.modality in ("text", "both")

    audio_cacher = (WavLMCacher(model_name=args.wavlm_model, device=args.device, layer_mode=args.layer_mode)
                    if do_audio else None)
    text_cacher = RoBERTaCacher(model_name=args.roberta_model, device=args.device) if do_text else None

    # optional speech_regions for the text-coverage cross-check / ami fallback
    regions = None
    if args.speech_regions:
        from src.data.speech_regions import SpeechRegions
        regions = SpeechRegions(args.speech_regions)
    corpus = None
    if args.text_source == "ami_words":
        from src.data.ami import AMICorpus
        corpus = AMICorpus(args.ami_root)

    clips_root = Path(args.clips_root)
    index_rows, errors, skipped_audio, coverage_warns = [], [], 0, 0

    for rec in pbar(records, desc=f"cache:{args.modality}"):
        sid = rec.sample_id
        try:
            audio_rel = text_rel = None

            if do_audio:
                if rec.missing_audio or not rec.human_mix_clip:
                    skipped_audio += 1
                    logger.info("skip audio (missing_audio) for %s", sid)
                else:
                    clip_path = clips_root / rec.human_mix_clip
                    wav = load_clip(clip_path, context_seconds=args.context_seconds, sample_rate=args.sample_rate)
                    feat = audio_cacher.encode(wav, num_bins=args.num_frames, bin_dur=args.frame_shift)
                    audio_rel = f"audio/{sid}.npy"
                    np.save(out_dir / audio_rel, feat.astype(np.float32))

            if do_text:
                if args.text_source == "manifest_events":
                    events = rec.human_text_events(rec.time)
                    # coverage cross-check vs speech_regions (warn if an active speaker is absent)
                    if regions is not None:
                        ev_spk = {e["speaker"] for e in events}
                        active = {sp for sp in rec.human_speakers
                                  if regions.regions_before(rec.meeting_id, sp, rec.time)
                                  and any(s < rec.time and e > rec.context_start
                                          for s, e in regions.regions_before(rec.meeting_id, sp, rec.time))}
                        missing = active - ev_spk
                        if missing:
                            coverage_warns += 1
                            logger.warning("%s: text_context missing speakers active in window: %s "
                                           "(consider --text-source ami_words)", sid, sorted(missing))
                    words = events_to_words(events)
                else:  # ami_words fallback
                    from src.features.text_roberta import collect_human_words
                    words = collect_human_words(corpus, rec.meeting_id, rec.human_speakers,
                                                rec.context_start, rec.time)
                feat = text_cacher.encode_sample(words, rec.context_start,
                                                 num_bins=args.num_frames, bin_dur=args.frame_shift)
                text_rel = f"text/{sid}.npy"
                np.save(out_dir / text_rel, feat.astype(np.float32))

            index_rows.append({"sample_id": sid, "split": rec.split,
                               "audio_path": audio_rel, "text_path": text_rel})
        except Exception as e:  # noqa: BLE001
            errors.append((sid, repr(e)))
            logger.error("%s: %s", sid, e)

    pd.DataFrame(index_rows).to_parquet(out_dir / "features_index.parquet", index=False)
    meta = {
        "wavlm_model": args.wavlm_model if do_audio else None,
        "roberta_model": args.roberta_model if do_text else None,
        "layer_mode": args.layer_mode if do_audio else None,
        "num_layers": audio_cacher.num_layers if do_audio else None,
        "audio_dim": audio_cacher.hidden_dim if do_audio else None,
        "text_dim": text_cacher.hidden_dim if do_text else None,
        "text_source": args.text_source,
        "num_frames": args.num_frames, "frame_shift": args.frame_shift,
        "context_seconds": args.context_seconds, "sample_rate": args.sample_rate,
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("indexed %d | audio skipped(missing): %d | coverage warns: %d | errors: %d -> %s",
                len(index_rows), skipped_audio, coverage_warns, len(errors), out_dir / "features_index.parquet")
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
    p.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    p.add_argument("--roberta-model", default="roberta-base")
    p.add_argument("--num-frames", type=int, default=120)
    p.add_argument("--frame-shift", type=float, default=0.05)
    p.add_argument("--context-seconds", type=float, default=6.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--strict", action="store_true")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
