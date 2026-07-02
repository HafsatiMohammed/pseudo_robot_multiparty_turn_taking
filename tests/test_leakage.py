#!/usr/bin/env python3
"""
Leakage boundary + causal-cut tests (runnable: `python tests/test_leakage.py`).

1. No feature builder may read anything under `future_for_labeling_only`. We load
   the fixture with the tripwire ON and run the full per-sample feature pipeline
   (timing from speech_regions, text from text_context, audio path resolution);
   any access of the label-only field would raise LeakageError -> test fails.
2. The tripwire actually fires when the field IS touched (positive control).
3. Causal cuts: timing regions use start < t; text events use end <= t.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.data.speech_regions import SpeechRegions
from src.data.timing_features import extract_sample
from src.eval.strata import speech_act_from_events
from src.features.text_roberta import events_to_words
from src.utils.manifest import LeakageError, load_records

FIX = _REPO / "tests" / "fixtures"


def test_no_leakage_and_causal_cuts():
    records = load_records(FIX / "manifest_sample.jsonl", tripwire_future=True)
    regions = SpeechRegions(FIX / "speech_regions.parquet")
    n_checked = 0
    for rec in records:
        t = rec.time
        # --- timing path (must not touch future) ---
        human_regions = regions.human_regions_before(rec.meeting_id, rec.human_speakers, t)
        robot_regions = regions.regions_before(rec.meeting_id, rec.pseudo_robot, t)
        # causal cut: every region started before t
        for regs in list(human_regions.values()) + [robot_regions]:
            for (s, _e) in regs:
                assert s < t, f"{rec.sample_id}: timing region onset {s} >= t {t}"
        extract_sample(human_regions, robot_regions, t=t, context_start=rec.context_start)

        # --- text path (must not touch future) ---
        events = rec.human_text_events(t)
        for e in events:
            assert e["end"] <= t, f"{rec.sample_id}: text event end {e['end']} > t {t}"
        events_to_words(events)
        speech_act_from_events(events, t)

        # --- audio path (must not touch future) ---
        _ = (rec.missing_audio, rec.human_mix_clip)
        n_checked += 1
    print(f"[ok] full feature pipeline ran on {n_checked} tripwired samples with NO leakage; "
          "timing start<t and text end<=t hold")


def test_tripwire_fires():
    rec = load_records(FIX / "manifest_sample.jsonl", tripwire_future=True)[0]
    for probe in (
        lambda: rec.future_for_labeling_only["target_did"],
        lambda: rec.get("future_for_labeling_only")["target_did"],
        lambda: list(rec.future_for_labeling_only),
    ):
        try:
            probe()
        except LeakageError:
            continue
        raise AssertionError("tripwire did NOT fire on future_for_labeling_only access")
    print("[ok] tripwire fires on every future_for_labeling_only access path")


if __name__ == "__main__":
    test_no_leakage_and_causal_cuts()
    test_tripwire_fires()
    print("\n*** LEAKAGE TESTS PASSED ***")
