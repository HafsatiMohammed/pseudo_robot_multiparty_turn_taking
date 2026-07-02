"""
The ONE place that knows the final_manifest.jsonl schema.

manifest_writer.py writes flat parquet columns EXCEPT three that are nested objects
expanded from JSON strings:
  - text_context              (text_context_json)              -> THE text source
  - state_at_prediction_time  (state_at_prediction_time_json)  -> input snapshot at t
  - future_for_labeling_only  (future_for_labeling_only_json)  -> LABEL-ONLY (leakage!)

`future_for_labeling_only` is a hard leakage boundary: no feature builder may read
anything under it. Load with tripwire_future=True to make any access raise.

All downstream manifest consumers (prepare_dataset, feature caching) go through
load_records / ManifestRecord so the schema lives in exactly one module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

NESTED_FIELDS = ("text_context", "state_at_prediction_time", "future_for_labeling_only")
LEAKAGE_KEY = "future_for_labeling_only"

# llm_* fields are EVALUATION/STRATA sources, never model inputs.
LLM_STRATA_FIELDS = {
    "completeness": "llm_current_human_speaker_complete",  # strata S4
    "floor_state": "llm_floor_state",                      # strata S5
    "social": "llm_would_entry_be_socially_reasonable",    # reserved (world-model reward)
}


class LeakageError(RuntimeError):
    """Raised when a feature builder touches future_for_labeling_only."""


class _FutureTripwire:
    """Wraps the label-only payload; ANY access raises LeakageError.

    Used by the leakage unit test: run the full feature pipeline with the future
    field tripwired and assert nothing raises -> nothing read it."""

    __slots__ = ("_payload",)

    def __init__(self, payload):
        object.__setattr__(self, "_payload", payload)

    def __getattr__(self, name):
        raise LeakageError(f"feature builder accessed {LEAKAGE_KEY}.{name}")

    def __getitem__(self, key):
        raise LeakageError(f"feature builder accessed {LEAKAGE_KEY}[{key!r}]")

    def __iter__(self):
        raise LeakageError(f"feature builder iterated {LEAKAGE_KEY}")

    def __contains__(self, key):
        raise LeakageError(f"feature builder probed {LEAKAGE_KEY}")

    def __repr__(self):
        return "<future_for_labeling_only: tripwired (label-only, no model access)>"


def expand_nested(d: dict) -> dict:
    """Re-expand the three nested objects if only their *_json string form is present."""
    for name in NESTED_FIELDS:
        if name not in d and f"{name}_json" in d:
            raw = d[f"{name}_json"]
            d[name] = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return d


class ManifestRecord:
    """Typed view over one manifest line. Feature builders read only allowed fields."""

    def __init__(self, data: dict, tripwire_future: bool = False):
        self._d = expand_nested(dict(data))
        fut = self._d.get(LEAKAGE_KEY) or {}
        self._future = _FutureTripwire(fut) if tripwire_future else fut
        if tripwire_future:
            # guard ALL access paths (.future_for_labeling_only AND .get(...))
            self._d[LEAKAGE_KEY] = self._future

    # --- core flat fields ---
    @property
    def sample_id(self) -> str: return self._d["sample_id"]
    @property
    def meeting_id(self) -> str: return self._d["meeting_id"]
    @property
    def pseudo_robot(self) -> str: return self._d["pseudo_robot"]
    @property
    def human_speakers(self) -> List[str]: return list(self._d.get("human_speakers") or [])
    @property
    def split(self) -> str: return self._d.get("split", "train")
    @property
    def time(self) -> float: return float(self._d["time"])
    @property
    def context_start(self) -> float: return float(self._d["context_start"])
    @property
    def context_end(self) -> float: return float(self._d["context_end"])
    @property
    def final_label(self): return self._d.get("final_label")

    # --- audio (pre-cut, human-only, already windowed) ---
    @property
    def human_mix_clip(self) -> Optional[str]: return self._d.get("human_mix_clip")
    @property
    def human_multichannel_clip(self) -> Optional[str]: return self._d.get("human_multichannel_clip")
    @property
    def missing_audio(self) -> bool: return bool(self._d.get("missing_audio", False))

    # --- nested objects ---
    @property
    def text_context(self) -> dict: return self._d.get("text_context") or {}
    @property
    def state_at_prediction_time(self) -> dict: return self._d.get("state_at_prediction_time") or {}
    @property
    def future_for_labeling_only(self):
        """Label-only payload (tripwired if loaded with tripwire_future=True)."""
        return self._future

    # --- llm_* (strata / analysis ONLY, never model inputs) ---
    def llm(self, key: str): return self._d.get(key)

    def get(self, key: str, default=None): return self._d.get(key, default)

    # --- text helpers (THE text source) ---
    def text_event_speakers(self) -> set:
        return {e.get("speaker") for e in (self.text_context.get("events") or [])}

    def human_text_events(self, until_t: float) -> List[dict]:
        """Human speakers' events with end <= until_t (causal text cut), time-ordered."""
        hs = set(self.human_speakers)
        out = [e for e in (self.text_context.get("events") or [])
               if e.get("speaker") in hs and e.get("text")
               and e.get("end") is not None and float(e["end"]) <= until_t]
        out.sort(key=lambda e: (float(e["start"]), float(e["end"])))
        return out


def iter_records(path, tripwire_future: bool = False) -> Iterator[ManifestRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Produced by the upstream groupturn_labeler pipeline (manifest_writer.py); "
            "NOT created here. For development use tests/fixtures/manifest_sample.jsonl."
        )
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON ({e})")
            yield ManifestRecord(data, tripwire_future=tripwire_future)


def load_records(path, tripwire_future: bool = False) -> List[ManifestRecord]:
    recs = list(iter_records(path, tripwire_future=tripwire_future))
    if not recs:
        raise ValueError(f"Manifest {path} contained no records.")
    return recs


# Back-compat: raw dicts (still expands nested objects).
def load_manifest(path) -> List[dict]:
    return [r._d for r in load_records(path)]
