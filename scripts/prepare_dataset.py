#!/usr/bin/env python3
"""
Phase 1a: build the timing-feature parquet from the manifest + upstream
speech_regions.parquet (oracle voice activity, word-merged at gap 0.25 s).

Activity for BOTH the timing model (X_frame, X_scalar) and the five VA-baseline
features comes from speech_regions.parquet (regions starting before t only), so
model activity and baseline activity share ONE source. Speaker membership comes
from the manifest's `human_speakers`. The manifest is read through the single
loader (src.utils.manifest); it is loaded with the future_for_labeling_only
tripwire ON, so any accidental read of label-only fields raises immediately.

Also passes through the strata sources (never model inputs):
  preceding_speech_act  (S3, from text_context)   llm_current_human_speaker_complete (S4)
  llm_floor_state (S5)

Output: train/validation/test.parquet, grouped by the manifest split (no re-split;
asserts no meeting spans splits), loadable by src.data.dataset.TimingDataset.

Usage:
    python scripts/prepare_dataset.py --manifest <final_manifest.jsonl> \
        --speech-regions <speech_regions.parquet> --output data/processed/timing
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.speech_regions import SpeechRegions
from src.data.timing_features import VA_FEATURES, extract_sample
from src.eval.strata import speech_act_from_events
from src.utils.manifest import load_records


def assert_split_integrity(records):
    """No meeting may span splits (all pseudo-robot views share one split)."""
    m2s = defaultdict(set)
    for r in records:
        m2s[r.meeting_id].add(r.split)
    bad = {m: sorted(s) for m, s in m2s.items() if len(s) > 1}
    if bad:
        raise ValueError(f"Meetings span multiple splits (not allowed): {bad}")


def process(args):
    records = load_records(args.manifest, tripwire_future=True)  # leakage tripwire ON
    if args.max_samples is not None:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} manifest record(s) from {args.manifest}")
    assert_split_integrity(records)

    regions = SpeechRegions(args.speech_regions)
    print(f"speech_regions: {args.speech_regions}")

    by_split = defaultdict(list)
    errors = []

    for rec in tqdm(records, desc="Extracting timing features"):
        sid = rec.sample_id
        try:
            t, cs, ce = rec.time, rec.context_start, rec.context_end
            if abs((ce - cs) - args.context_seconds) > 1e-3:
                print(f"  [warn] {sid}: window {ce - cs:.3f}s != {args.context_seconds}s")

            humans = rec.human_speakers
            human_regions = regions.human_regions_before(rec.meeting_id, humans, t)
            robot_regions = regions.regions_before(rec.meeting_id, rec.pseudo_robot, t)

            x_frame, x_scalar, va = extract_sample(
                human_regions, robot_regions, t=t, context_start=cs,
                num_frames=args.num_frames, frame_shift=args.frame_shift,
                context_seconds=args.context_seconds,
            )

            row = {
                "sample_id": sid, "meeting_id": rec.meeting_id,
                "pseudo_robot": rec.pseudo_robot, "time": t,
                "context_start": cs, "context_end": ce, "split": rec.split,
                "X_frame": x_frame.astype(np.float32).tolist(),
                "X_scalar": x_scalar.astype(np.float32).tolist(),
                "final_label": rec.final_label,
                "weak_label": rec.get("weak_label"),
                "llm_confidence": rec.get("llm_confidence"),
                "num_humans": len(human_regions),
                # strata sources (NEVER model inputs)
                "preceding_speech_act": speech_act_from_events(
                    rec.human_text_events(t), t),                       # S3 (text)
                "llm_current_human_speaker_complete": rec.llm("llm_current_human_speaker_complete"),  # S4
                "llm_floor_state": rec.llm("llm_floor_state"),          # S5
            }
            for k in VA_FEATURES:
                row[k] = va[k]
            by_split[rec.split].append(row)
        except Exception as e:  # noqa: BLE001
            errors.append((sid, repr(e)))
            if len(errors) <= 10:
                print(f"  [error] {sid}: {e}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Summary ===")
    total_ok = 0
    for split in ("train", "validation", "test"):
        rows = by_split.get(split, [])
        if not rows:
            print(f"{split:11s}: 0 samples (skipped)")
            continue
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
        total_ok += len(df)
        print(f"{split:11s}: {len(df):5d} -> {out_dir/f'{split}.parquet'}  labels={dict(Counter(df['final_label']))}")

    print(f"\nOK: {total_ok}   Errors: {len(errors)}")
    if errors and args.strict:
        raise SystemExit(f"{len(errors)} sample(s) failed and --strict is set.")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="final_manifest.jsonl")
    p.add_argument("--speech-regions", required=True, help="speech_regions.parquet (upstream oracle VA)")
    p.add_argument("--output", required=True, help="Output dir for parquet files.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--frame-shift", type=float, default=0.05)
    p.add_argument("--num-frames", type=int, default=120)
    p.add_argument("--context-seconds", type=float, default=6.0)
    p.add_argument("--strict", action="store_true")
    return p


if __name__ == "__main__":
    process(build_argparser().parse_args())
