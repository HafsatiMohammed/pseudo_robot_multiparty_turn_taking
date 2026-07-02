#!/usr/bin/env python3
"""
Dev tool (NOT part of the pipeline): regenerate the fixture in the REAL manifest
shape from the local AMI corpus, so downstream phases can be developed/verified
without the upstream artifacts.

Produces:
  tests/fixtures/manifest_sample.jsonl   real-shape lines (flat + 3 nested objects)
  tests/fixtures/speech_regions.parquet  oracle word-merged regions (gap 0.25 s)
  tests/fixtures/clips/<sample_id>.wav   pre-cut human-only mono mix clips

The nested `future_for_labeling_only` block is deliberately filled with post-t info
(the leakage "poison") so the leakage test can confirm no feature builder reads it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.data.ami import AMICorpus
from src.data.regions import merge_words_to_regions
from src.features.audio_wavlm import build_group_audio

AMI_ROOT = "/home/mohammed/Documents/robot-group-turn-taking-labeler/data/ami_original"
GAP = 0.25
CTX = 6.0
FIX = _REPO / "tests" / "fixtures"
CLIPS = FIX / "clips"

# (meeting, robot, t, split, final_label, missing_audio, llm_floor_state, llm_complete)
SPECS = [
    ("ES2002a", "A", 100.0, "train", "WAIT", False, "HUMAN_HOLDING_FLOOR", "YES"),
    ("ES2002a", "B", 150.0, "train", "START_SPEAKING", False, "FLOOR_OPEN", "YES"),
    ("ES2002a", "C", 200.0, "train", "BACKCHANNEL", False, "HUMAN_HOLDING_FLOOR", "NO"),
    ("ES2002a", "D", 250.0, "train", "WAIT", True, "FLOOR_OPEN", "YES"),     # missing_audio
    ("IS1000a", "A", 120.0, "validation", "WAIT", False, "HUMAN_HOLDING_FLOOR", "NO"),
    ("IS1000a", "D", 180.0, "validation", "BACKCHANNEL", False, "FLOOR_OPEN", "YES"),
    ("EN2001a", "E", 130.0, "test", "WAIT", False, "HUMAN_HOLDING_FLOOR", "YES"),
    ("EN2001a", "A", 200.0, "test", "START_SPEAKING", False, "FLOOR_OPEN", "NO"),
]


def main():
    corpus = AMICorpus(AMI_ROOT)
    CLIPS.mkdir(parents=True, exist_ok=True)

    # 1. speech_regions.parquet for the fixture meetings (all speakers, gap 0.25)
    reg_rows = []
    for meeting in sorted({s[0] for s in SPECS}):
        for letter in corpus.get_meeting(meeting).letters:
            words = corpus.read_words(meeting, letter)
            for (s, e) in merge_words_to_regions([w for w in words], gap_threshold=GAP):
                reg_rows.append({"meeting_id": meeting, "speaker": letter, "start": s, "end": e})
    pd.DataFrame(reg_rows).to_parquet(FIX / "speech_regions.parquet", index=False)
    print(f"speech_regions.parquet: {len(reg_rows)} regions")

    # 2. per-sample manifest lines
    lines = []
    for (meeting, robot, t, split, label, missing, floor, complete) in SPECS:
        cs, ce = t - CTX, t
        sid = f"{meeting}__{robot}__{int(round(t*1000)):08d}"
        agents = corpus.get_meeting(meeting).letters
        humans = [a for a in agents if a != robot]

        # text_context: ALL human speakers' word tokens overlapping the window
        events = []
        for h in humans:
            for w in corpus.read_words(meeting, h):
                if w.start >= cs and w.start < ce:
                    events.append({"speaker": h, "text": w.text, "start": round(w.start, 3),
                                   "end": round(w.end, 3), "is_punc": bool(w.is_punc)})
        events.sort(key=lambda e: (e["start"], e["end"]))

        # audio clip (pre-cut, human-only mono mix) unless missing
        clip_rel = None
        if not missing:
            wav = build_group_audio(corpus, meeting, humans, cs, context_seconds=CTX, sample_rate=16000)
            clip_path = CLIPS / f"{sid}.wav"
            sf.write(clip_path, wav.astype(np.float32), 16000, subtype="PCM_16")
            clip_rel = f"tests/fixtures/clips/{sid}.wav"

        # future_for_labeling_only: post-t info (LEAKAGE POISON; never a model input)
        future_words = [{"speaker": h, "text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
                        for h in (humans + [robot]) for w in corpus.read_words(meeting, h)
                        if w.start >= t and w.start < t + 2.0][:20]
        future = {"target_did": label, "target_window": [t, t + 2.0], "future_words": future_words}

        rec = {
            "sample_id": sid, "meeting_id": meeting, "pseudo_robot": robot,
            "human_speakers": humans, "split": split,
            "time": t, "context_start": cs, "context_end": ce,
            "prediction_start": t, "prediction_end": t + 2.0, "prediction_horizon": 2.0,
            "weak_label": label, "wait_subtype": None,
            "entry_subtype": ("backchannel" if label == "BACKCHANNEL" else
                              ("turn" if label == "START_SPEAKING" else None)),
            "llm_entry_type": ("BACKCHANNEL" if label == "BACKCHANNEL" else
                               ("START" if label == "START_SPEAKING" else "NONE")),
            "llm_floor_state": floor,
            "llm_floor_was_open": (floor == "FLOOR_OPEN"),
            "llm_would_entry_be_socially_reasonable": (label != "WAIT"),
            "llm_current_human_speaker_complete": complete,
            "llm_confidence": 0.8, "llm_reason": "fixture-synthetic",
            "final_label": label,
            "human_mix_clip": clip_rel,
            "human_multichannel_clip": None,
            "human_audio_tracks_json": json.dumps([]),
            "missing_audio": missing,
            # nested objects (as the writer emits them: real objects)
            "text_context": {"events": events},
            "state_at_prediction_time": {"note": "snapshot; features recomputed from speech_regions"},
            "future_for_labeling_only": future,
        }
        lines.append(json.dumps(rec))

    (FIX / "manifest_sample.jsonl").write_text("\n".join(lines) + "\n")
    print(f"manifest_sample.jsonl: {len(lines)} samples; clips: {len(list(CLIPS.glob('*.wav')))}")


if __name__ == "__main__":
    main()
