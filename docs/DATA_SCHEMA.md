# DATA_SCHEMA.md

Data contract for the pseudo-robot group turn-taking pipeline, aligned to the **real**
`final_manifest.jsonl` produced by the upstream `groupturn_labeler` pipeline
(`manifest_writer.py`, scripts 01–14). The single module that knows this schema is
[`src/utils/manifest.py`](../src/utils/manifest.py); `prepare_dataset` and feature caching
read the manifest only through it.

---

## 0. Upstream artifacts this pipeline consumes (do not recompute)

| Artifact | What it is | Used for |
|---|---|---|
| `final_manifest.jsonl` | one decision sample per line (below) | sample definitions, text, labels, audio paths |
| `speech_regions.parquet` | oracle voice activity, word-merged at **gap 0.25 s** | **all** activity → timing features **and** the 5 VA-baseline features |
| `speaker_channel_map.parquet` | speaker↔headset map | (upstream only; not needed at runtime here) |
| pre-cut human-only audio clips | `human_mix_clip` (mono mix, robot excluded, already windowed) | audio branch (no headset re-mixing) |

Dev note: none of these exist locally yet. [`tests/build_fixture.py`](../tests/build_fixture.py)
regenerates a tiny real-shape fixture (`tests/fixtures/manifest_sample.jsonl`,
`speech_regions.parquet`, `clips/*.wav`) from the local AMI corpus to develop against.

`speech_regions.parquet` schema: `meeting_id, speaker, start, end` (one row per merged region).

---

## 1. Manifest record (one JSON object per line)

`manifest_writer.py` writes all columns **flat** EXCEPT three that are **nested objects**
(expanded from their `*_json` string forms). The loader handles either form.

### Flat fields
| Field | Meaning / use |
|---|---|
| `sample_id` | unique key; names every cached artifact |
| `meeting_id` | AMI observation id; joins to `speech_regions` + audio clips |
| `pseudo_robot` | masked participant letter |
| `human_speakers` | **explicit list** of the human letters (no group-size guessing) |
| `split` | `train`/`validation`/`test` — **already assigned; never re-split** |
| `time` (`t`), `context_start`, `context_end` | decision point + 6 s window (`t == context_end`) |
| `prediction_start` / `prediction_end` / `prediction_horizon` | label-derivation horizon (post-`t`); **not model inputs** |
| `weak_label`, `wait_subtype`, `entry_subtype` | pre/aux labels (diagnostics) |
| `llm_*` (`llm_entry_type`, `llm_floor_state`, `llm_floor_was_open`, `llm_would_entry_be_socially_reasonable`, `llm_current_human_speaker_complete`, `llm_confidence`, `llm_reason`, …) | **EVALUATION/STRATA ONLY — never model inputs** (see §5) |
| `final_label` | the target: `WAIT`/`BACKCHANNEL`/`START_SPEAKING` |
| `human_mix_clip` | path to the pre-cut mono human-mix clip (audio source) |
| `human_multichannel_clip` | optional per-speaker stack (if multichannel ever needed) |
| `human_audio_tracks_json` | per-speaker track paths (JSON string) |
| `missing_audio` | bool — if true, **skip audio** for this sample (logged) |

### Nested objects (expanded from `*_json`)
| Object | Contents | Use |
|---|---|---|
| `text_context` | `{events: [{speaker, text, start, end, is_punc?}, …]}` — per-speaker word/utterance events for the window | **THE text source** (RoBERTa). Cut at `end ≤ t`. |
| `state_at_prediction_time` | snapshot of inputs available at `t` | reference; features are recomputed from `speech_regions` |
| `future_for_labeling_only` | fields used ONLY to derive labels | **HARD LEAKAGE BOUNDARY — never read by any feature builder** (§4) |

---

## 2. What `prepare_dataset.py` emits (Phase 1a)

One parquet per split (grouped by manifest `split`; asserts **no meeting spans splits**),
loadable by `TimingDataset`:

| Column | Notes |
|---|---|
| `sample_id, meeting_id, pseudo_robot, time, context_start, context_end, split` | passthrough |
| `X_frame` float32 `[120,7]`, `X_scalar` `[6]` | timing features (§3) from `speech_regions` |
| 5 VA cols: `human_active_at_t, num_humans_active_at_t, overlap_active_at_t, silence_duration_before_t, current_human_speech_duration` | from the **same** `speech_regions` |
| `final_label`, `weak_label`, `llm_confidence`, `num_humans` | passthrough/diagnostic |
| **strata sources** (never model inputs): `preceding_speech_act` (S3, from `text_context`), `llm_current_human_speaker_complete` (S4), `llm_floor_state` (S5) | precomputed so strata reads only columns |

The 7 frame + 6 scalar feature definitions are unchanged (see
[`src/data/timing_features.py`](../src/data/timing_features.py)); only the **activity
source** changed (AMI words XML → `speech_regions.parquet`).

---

## 3. Frame grid & feature math

50 ms / 20 Hz, context `[t−6 s, t]` → exactly **120 frames**. Frame features over the
human group (`human_speakers`); `num_humans_active_norm` denominator = `len(human_speakers)`.
Audio mean-pooled into the 120 bins; text events placed by `[start,end]`.

---

## 4. Leakage boundary & causal cuts (enforced + tested)

- **By name:** no feature builder may read anything under `future_for_labeling_only`.
  `prepare_dataset` and `cache_features` load the manifest with `tripwire_future=True`, so
  any access (`.future_for_labeling_only` **or** `.get("future_for_labeling_only")`) raises
  `LeakageError`. [`tests/test_leakage.py`](../tests/test_leakage.py) runs the full feature
  pipeline tripwired (no access) and confirms the tripwire fires when touched.
- **Causal asymmetry (documented once):**
  - **timing** uses regions with **onset `< t`** (a region may extend past `t` — speech ongoing at the decision point is observed).
  - **text & audio** use **`end ≤ t`** (only fully-uttered words / a clip ending at `t`) — stricter, because lexical/acoustic content completed after `t` would leak.

---

## 5. Labels & the `llm_*` fields

Target = `final_label` ∈ {WAIT, BACKCHANNEL, START_SPEAKING}; `entry = {BC, START}`.
`llm_*` are **diagnostic/strata sources only, never model features**:
`llm_current_human_speaker_complete` → **S4**, `llm_floor_state` → **S5**,
`llm_would_entry_be_socially_reasonable` → reserved (world-model reward, later).

---

## 6. Artifacts THIS pipeline adds (Phase 1 cache)

Per `sample_id`, under the cache dir, with `features_index.parquet`
(`sample_id → {audio_path, text_path}`, relative) + `cache_meta.json`:

| Artifact | Shape | Source | Encoder (frozen, offline, eval/no_grad) |
|---|---|---|---|
| audio | `[120,768]` default (fixed layer-3–8 weighted collapse; `[120,13,768]` under `--layer-mode all`) | `human_mix_clip` (loaded, resampled to 16 kHz, fixed 6 s; **skipped if `missing_audio`** → zeros at train time) | `microsoft/wavlm-base-plus` (~94M, frozen) |
| text | `[120,768]` | `text_context` events (human, `end ≤ t`); AMI words only behind `--text-source ami_words` | `roberta-base` (~125M, frozen) |

Encoders are never instantiated or backpropped in training (`frozen=0` asserted). A
text-coverage cross-check warns if a speaker active in the window (per `speech_regions`) is
absent from `text_context` (→ consider the AMI-words fallback).

---

## 7. Resolved schema decisions

- **human_speakers** is explicit in the manifest → used directly (no AMI/group-size guessing).
- **`time == context_end`**; `prediction_*` is the post-`t` horizon, never a model input.
- **Audio** = pre-cut `human_mix_clip` (no headset re-mixing, no runtime channel map).
- **Text** = `text_context` (default); AMI words behind a switch.
- **Timing + VA** = `speech_regions.parquet` (one shared activity source).
- **Splits** are upstream; grouped, integrity-asserted, never re-split.

---

## 8. Label space & sample-id

`WAIT`=0, `BACKCHANNEL`=1, `START_SPEAKING`=2 (matches `TimingDataset.LABEL_TO_IDX`).
`sample_id` is an opaque, filesystem-safe key (fixture form `<meeting>__<robot>__<t_ms:08d>`).

> **Loader fixes to `TimingDataset` (still required):** parquet object-array cell coercion
> and per-feature normalization stats (no `keepdims`) — see `src/data/dataset.py`.
