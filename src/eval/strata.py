"""
Phase 6: per-stratum (diagnostic) evaluation. Each stratum splits the test set in
two; we report the full metric suite per slice for every system + Delta-vs-timing.

Stratum sources are PRECOMPUTED by prepare_dataset into the timing parquet (so the
manifest schema stays in one place and strata reads only columns):
  S1 Pause length  : silence_duration_before_t >= theta (0.6 s) vs <    [control: timing]
  S2 Overlap       : overlap_active_at_t True vs False                  [isolates audio]
  S3 Speech act    : preceding human utterance is a question vs statement
                     (from text_context; ? / wh-word / aux-inversion)   [isolates text]
  S4 Completeness  : llm_current_human_speaker_complete == NO vs YES     [text; WAIT-ish]
  S5 Floor state   : llm_floor_state HUMAN_HOLDING_FLOOR vs FLOOR_OPEN   [text+audio]

llm_* fields are EVALUATION-ONLY (never model inputs). Strata are binary; do NOT
cross strata. Min cell size (~50): below it, report only false/missed-entry.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_all, false_entry_rate, missed_entry_rate

WH_AUX = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "do", "does", "did", "is", "are", "am", "was", "were", "can", "could",
    "would", "will", "shall", "should", "may", "might", "have", "has", "had",
}
TERMINALS = {"?", ".", "!"}


# ---------------------------------------------------------------------------
# S3 — speech act from manifest text_context events (THE text source)
# ---------------------------------------------------------------------------
def speech_act_from_events(events: List[dict], t: float) -> Optional[str]:
    """
    Classify the last human utterance ending before t as 'question'/'statement'.
    Question if it ends with '?', else if its first word is wh / auxiliary.
    `events` = text_context events (already human-only, end <= t), time-ordered.
    Returns None if there are no events.
    """
    toks = [e for e in events if e.get("text")]
    if not toks:
        return None
    term_idx = [i for i, e in enumerate(toks)
                if e.get("is_punc") and str(e["text"]).strip() in TERMINALS]
    if term_idx:
        last = term_idx[-1]
        prev = term_idx[-2] if len(term_idx) >= 2 else -1
        utter = toks[prev + 1: last + 1]
        if str(toks[last]["text"]).strip() == "?":
            return "question"
    else:
        utter = toks
    first = [str(e["text"]).strip().lower() for e in utter
             if not e.get("is_punc") and str(e["text"]).strip()]
    if first and first[0] in WH_AUX:
        return "question"
    return "statement"


# ---------------------------------------------------------------------------
# Tags from the precomputed timing-parquet columns
# ---------------------------------------------------------------------------
def tag_samples(test_df: pd.DataFrame, pause_theta: float = 0.6) -> pd.DataFrame:
    def _opt_bool(series, true_val):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else (v == true_val) for v in series]

    out = pd.DataFrame({"sample_id": test_df["sample_id"].to_numpy()})
    out["pause_long"] = (test_df["silence_duration_before_t"].astype(float) >= pause_theta).to_numpy()
    out["overlap"] = test_df["overlap_active_at_t"].astype(bool).to_numpy()
    if "preceding_speech_act" in test_df:
        out["is_question"] = _opt_bool(test_df["preceding_speech_act"], "question")
    else:
        out["is_question"] = [None] * len(test_df)
    out["incomplete"] = (_opt_bool(test_df["llm_current_human_speaker_complete"], "NO")
                         if "llm_current_human_speaker_complete" in test_df else [None] * len(test_df))
    out["floor_holding"] = (_opt_bool(test_df["llm_floor_state"], "HUMAN_HOLDING_FLOOR")
                            if "llm_floor_state" in test_df else [None] * len(test_df))
    return out


# (id, label, column, (sliceA_label, value), (sliceB_label, value))
STRATA = [
    ("S1_pause", "Pause length", "pause_long", ("long", True), ("short", False)),
    ("S2_overlap", "Overlap", "overlap", ("overlap", True), ("clean", False)),
    ("S3_speechact", "Preceding speech act", "is_question", ("question", True), ("statement", False)),
    ("S4_completeness", "Syntactic completeness", "incomplete", ("incomplete", True), ("complete", False)),
    ("S5_floorstate", "Floor state", "floor_holding", ("holding", True), ("open", False)),
]


def _slice_metrics(y_true, y_pred, probs, mask, min_cell: int) -> Dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    pr = probs[mask] if probs is not None else None
    if n < min_cell:
        return {"n": n, "small_cell": True,
                "false_entry": false_entry_rate(yt, yp),
                "missed_entry": missed_entry_rate(yt, yp)}
    m = compute_all(yt, yp, pr)
    m["small_cell"] = False
    return m


def stratified_metrics(
    tags: pd.DataFrame, preds_by_system: Dict[str, Dict],
    min_cell: int = 50, timing_key: str = "timing",
) -> Dict:
    """preds_by_system[name] = {"sample_id":[...], "y_true":arr, "y_pred":arr, "probs":arr|None}"""
    out = {}
    for sid_, label, col, (la, va), (lb, vb) in STRATA:
        tag_map = dict(zip(tags["sample_id"], tags[col]))
        stratum = {"label": label, "column": col, "slices": {}, "delta_macro_f1": {}}
        for slabel, sval in ((la, va), (lb, vb)):
            stratum["slices"][slabel] = {}
            timing_mf1 = None
            for name, P in preds_by_system.items():
                ids = np.asarray(P["sample_id"])
                mask = np.array([tag_map.get(s) == sval for s in ids])
                m = _slice_metrics(np.asarray(P["y_true"]), np.asarray(P["y_pred"]),
                                   P.get("probs"), mask, min_cell)
                stratum["slices"][slabel][name] = m
                if name == timing_key and not m.get("small_cell", True):
                    timing_mf1 = m.get("macro_f1")
            deltas = {}
            for name, m in stratum["slices"][slabel].items():
                if timing_mf1 is not None and m.get("macro_f1") is not None:
                    deltas[name] = m["macro_f1"] - timing_mf1
            stratum["delta_macro_f1"][slabel] = deltas
        out[sid_] = stratum
    return out
