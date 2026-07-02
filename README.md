# GroupTurn-Fuse — pseudo-robot multi-party turn-taking

At each moment a masked meeting participant (the "pseudo-robot") must **WAIT**,
**BACKCHANNEL**, or **START_SPEAKING** from the human group's recent context. We compare
**timing**, **acoustic** (WavLM), and **lexical** (RoBERTa) signals under a
**capacity-matched** ablation — the contribution is the *formulation + diagnostic*, not
"fusion improves F1". Built for ICASSP 2027.

> **Read [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) first** — it is the authoritative data
> contract (manifest record, AMI→features mapping, frame grid, label space, cached
> artifacts, and the resolved design decisions).

## Status

- The upstream **`final_manifest.jsonl` does not exist yet** (separate Qwen-validation
  pipeline). All scripts hard-fail if it is absent and are developed against the synthetic
  fixture `tests/fixtures/manifest_sample.jsonl`. Every phase below has been **verified
  end-to-end on that fixture** (shapes, causality, matched capacity, ablation enforcement,
  resume identity, exact metric values). The numbers are degenerate on 8 synthetic samples;
  real numbers come when the manifest lands — just point `--manifest` at it.

## Environment

`base` conda env, extended with the project deps (Python 3.13):

```bash
python -m pip install -r requirements.txt
```

This repo was validated with: torch 2.9 (CPU), torchaudio, transformers 5.12, pandas 3.0,
pyarrow 24, scikit-learn 1.9, librosa 0.11, matplotlib, soundfile. Use that interpreter for
all commands below (shown as `python`). A GPU is optional — pass `--device cuda`.

## Non-negotiables (enforced in code)

- **Frozen encoders, offline-cached.** WavLM / RoBERTa run once in `eval()`/`no_grad`,
  written to disk; training never instantiates or backprops them.
- **Matched capacity.** All four systems are the *same* `GroupTurnFuse` network; disabled
  modalities are zeroed at branch input (verified: identical parameter counts; timing-only
  logits invariant to audio/text swaps).
- **No leakage / past-only.** Features use only information with timestamp `< t` (timing:
  words with `start < t`; text: words with `end <= t`; audio clip ends at `t`).
- **Train balanced, evaluate natural.** Weighted sampler on train only; test on its natural
  (WAIT-dominated) distribution.
- **Tune on validation only; ≥3 seeds; deterministic + resumable.**

## Paths (set in `configs/base.yaml`, never hardcoded in code)

- `ami_root`: local AMI corpus (`headset/<meeting>/audio/...`, `ami_manual_1.6.1/words/...`).
- `manifest`: upstream `final_manifest.jsonl` (pending). Use the fixture meanwhile.

## Pipeline (exact CLI)

Every script supports `--max-samples N` for fast dry runs.

```bash
MAN=tests/fixtures/manifest_sample.jsonl              # swap for the real manifest
REGIONS=tests/fixtures/speech_regions.parquet         # upstream oracle VA
CLIPS=.                                                # root for human_mix_clip paths

# (dev only) regenerate the real-shape fixture from local AMI:
# python tests/build_fixture.py

# Phase 1a — timing features + 5 VA features + splits -> parquet (from speech_regions)
python scripts/prepare_dataset.py --manifest $MAN --speech-regions $REGIONS \
    --output data/processed/timing

# Phase 1 — cache frozen WavLM (from human_mix_clip) + RoBERTa (from text_context)
# --layer-mode sum collapses the WavLM layer axis with a FIXED (not learned) weighting
# concentrated on layers 3-8 (lower-middle) -> [120,768] per sample (~150 GB cache).
# Omit --layer-weights for the default layer 3-8 mean; pass an explicit 13-vector to
# override (normalized in code). Use --layer-mode all only to keep every layer ([120,13,768]).
python -m scripts.cache_features.run --manifest $MAN --output data/processed/cache \
    --modality both --clips-root $CLIPS --speech-regions $REGIONS --layer-mode sum \
    --layer-weights 0,0,0,1,1,1,1,1,1,0,0,0,0

# Phase 4 — non-learned baselines (majority, VA-Silence, VA-Threshold) + DET/EER
python scripts/eval_baselines.py --timing-dir data/processed/timing \
    --output reports/baselines

# Phase 5 — train one system + one seed (deterministic, resumable)
python scripts/train.py --base configs/base.yaml --system timing --seed 42 \
    --timing-dir data/processed/timing --cache-dir data/processed/cache \
    --out reports/runs/timing_seed42
#   resume: add --resume reports/runs/timing_seed42/last.ckpt

# Phase 7 — train all 4 systems x N seeds + assemble tables/figures (MD + LaTeX)
python scripts/run_ablation.py --base configs/base.yaml \
    --timing-dir data/processed/timing --cache-dir data/processed/cache \
    --runs-dir reports/runs --output reports/ablation --seeds 13 21 42
```

Systems: `timing`, `audio_timing`, `text_timing`, `full`.

## Layout

```
configs/         base.yaml (+ paths, model arch, train) + per-system yamls
scripts/         prepare_dataset.py, cache_features/, eval_baselines.py, train.py, run_ablation.py
src/data/        ami.py, regions.py, timing_features.py, dataset.py, loaders.py, multimodal.py
src/features/    align.py, audio_wavlm.py, text_roberta.py   (frozen-encoder caching)
src/models/      branches.py, models_multimodal.py           (GroupTurnFuse — locked design)
src/baselines/   rules.py, va_baseline_fixes.py
src/eval/        metrics.py, strata.py, tables.py, figures.py
src/utils/       config.py, seed.py, checkpoint.py, manifest.py
docs/            DATA_SCHEMA.md
reports/         runs/<system>_seed<seed>/ (ckpts, metrics.csv, probs_*.npz), ablation/, baselines/
tests/fixtures/  manifest_sample.jsonl
```

## Metrics & diagnostic (per `docs/` spec)

Macro-F1 (headline), per-class F1, false-entry, missed-entry, DET/EER (1−P(WAIT) swept),
action-type errors, ECE, bootstrap CI (full vs best single-modality). Core-3 strata:
**S1 pause length** (control, timing), **S2 overlap** (audio), **S3 preceding speech act**
(text), each split binary with a ≥50 cell-size guard and Δ-vs-timing-only.
