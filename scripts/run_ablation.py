#!/usr/bin/env python3
"""
Phase 7: ablation runner.

  1. Train all 4 systems x N seeds (resume-aware; skips completed runs, resumes a
     partial last.ckpt).
  2. Run the non-learned baselines (majority, VA-Silence, VA-Threshold).
  3. Assemble:
       - Main table (mean +/- std over seeds, best per column bold) -> MD + LaTeX.
       - Stratified tables S1-S3 (diagnostic) -> MD.
       - Figures: DET (all systems incl. rules), confusion matrices, gate-inspection.
     under --output (reports/ablation/).

Usage:
    python scripts/run_ablation.py --base configs/base.yaml \
        --timing-dir data/processed/timing --cache-dir data/processed/cache \
        --runs-dir reports/runs --output reports/ablation \
        --seeds 13 21 42 [--epochs N] [--max-samples M] [--skip-train]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.multimodal import get_multimodal_dataloaders
from src.eval import figures
from src.eval.strata import STRATA, stratified_metrics, tag_samples
from src.eval.tables import (
    MAIN_COLS, MAIN_ROWS, TRAINED, aggregate, render_main_latex, render_main_markdown,
    render_stratified_markdown,
)
from src.models.models_multimodal import build_all_systems, build_system
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging_setup import pbar, setup_logging
from src.utils.params import assert_frozen_zero, params_summary

logger = logging.getLogger(__name__)

DISPLAY_TRAINED = {"timing": "Timing-only", "audio_timing": "Audio + timing",
                   "text_timing": "Text + timing", "full": "GroupTurn-Fuse (full)"}


def write_model_params(cfg, out: Path) -> dict:
    """Build all four systems, assert matched capacity + frozen==0, persist params."""
    systems = build_all_systems({"model": cfg["model"]})
    summ = {}
    for name, m in systems.items():
        assert_frozen_zero(m)
        s = params_summary(m)
        summ[name] = s
        logger.info("params | system=%s | total=%d | active=%d | trainable=%d | frozen=%d",
                    name, s["total"], s["active"], s["trainable"], s["frozen"])
    totals = {n: s["total"] for n, s in summ.items()}
    assert len(set(totals.values())) == 1, f"Matched-capacity violated: {totals}"
    logger.info("Matched capacity OK: all systems = %d params", next(iter(totals.values())))
    with open(out / "model_params.json", "w") as f:
        json.dump(summ, f, indent=2)
    lines = ["# Model parameters (matched capacity)\n",
             "| system | total params | active params | trainable | frozen |",
             "|---|---|---|---|---|"]
    for name in ("timing", "audio_timing", "text_timing", "full"):
        s = summ[name]
        lines.append(f"| {name} | {s['total']:,} | {s['active']:,} | {s['trainable']:,} | {s['frozen']} |")
    (out / "model_params.md").write_text("\n".join(lines) + "\n")
    return summ

SYSTEMS = ["timing", "audio_timing", "text_timing", "full"]
DISPLAY = {k: n for k, n in MAIN_ROWS}


def run_training(system, seed, args):
    run_dir = Path(args.runs_dir) / f"{system}_seed{seed}"
    final = run_dir / "final_metrics.json"
    probs = run_dir / "probs_test.npz"
    if final.exists() and probs.exists():
        logger.info("[skip] %s seed%d (complete)", system, seed)
        return run_dir
    cmd = [sys.executable, "scripts/train.py", "--base", args.base, "--system", system,
           "--seed", str(seed), "--timing-dir", args.timing_dir, "--cache-dir", args.cache_dir,
           "--out", str(run_dir), "--device", args.device, "--num-workers", str(args.num_workers),
           "--cache-format", args.cache_format, "--log-level", args.log_level]
    if args.epochs is not None:
        cmd += ["--max-epochs", str(args.epochs)]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.amp:
        cmd += ["--amp"]
    last = run_dir / "last.ckpt"
    if last.exists():
        cmd += ["--resume", str(last)]
        logger.info("[resume] %s seed%d", system, seed)
    else:
        logger.info("[train ] %s seed%d", system, seed)
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
    return run_dir


def run_cotrain(args):
    """Co-train all 4 systems on ONE shared batch stream per seed (scripts/train_cotrain.py).

    Produces the SAME per-run outputs as the sequential path (reports/runs/<system>_seed<seed>/)
    but reads the feature cache once per seed instead of once per system -- ~4x less disk I/O.
    Verified bit-identical to the sequential runs (scripts/verify_cotrain_equivalence.py).
    --resume is passed so complete seeds are skipped and partial ones resume, matching the
    resume-aware behavior of run_training."""
    cmd = [sys.executable, "scripts/train_cotrain.py", "--base", args.base,
           "--timing-dir", args.timing_dir, "--cache-dir", args.cache_dir,
           "--runs-dir", args.runs_dir, "--device", args.device,
           "--num-workers", str(args.num_workers), "--cache-format", args.cache_format,
           "--log-level", args.log_level, "--seeds", *[str(s) for s in args.seeds], "--resume"]
    if args.epochs is not None:
        cmd += ["--max-epochs", str(args.epochs)]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.amp:
        cmd += ["--amp"]
    logger.info("[co-train] all systems x %d seed(s) on a shared batch stream", len(args.seeds))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def run_baselines(args, out):
    bdir = out / "baselines"
    cmd = [sys.executable, "scripts/eval_baselines.py", "--timing-dir", args.timing_dir,
           "--output", str(bdir)]
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
    return json.loads((bdir / "baseline_metrics.json").read_text()), bdir


def load_probs(run_dir):
    z = np.load(run_dir / "probs_test.npz", allow_pickle=True)
    return {"sample_id": list(z["sample_id"]), "y_true": z["y_true"],
            "y_pred": z["y_pred"], "probs": z["probs"]}


def gate_figure(args, cfg, runs_dir, tags, out):
    """Mean modality gate of the FULL system within each stratum slice."""
    full_seed = args.seeds[0]
    best = Path(runs_dir) / f"full_seed{full_seed}" / "best.ckpt"
    if not best.exists():
        logger.warning("no full-system best.ckpt; skipping gate figure.")
        return
    dls = get_multimodal_dataloaders(args.timing_dir, args.cache_dir, batch_size=64,
                                     num_workers=0, use_weighted_sampler=False,
                                     max_samples=args.max_samples)
    model = build_system("full", {"model": cfg["model"]})
    model.load_state_dict(load_checkpoint(best)["model"])
    model.eval()
    import torch
    ids, gates = [], []
    with torch.no_grad():
        for b in dls["test"]:
            _, g = model(b["frame"], b["scalar"], b["audio"], b["text"], return_gates=True)
            gates.append(g.cpu().numpy()); ids.extend(b["sample_id"])
    gates = np.concatenate(gates) if gates else np.zeros((0, 4))
    gid = {s: gates[i] for i, s in enumerate(ids)}
    modalities = ["timing", "scalar", "audio", "text"]
    by_slice = {}
    for sid, _label, col, (la, va), (lb, vb) in STRATA:
        tmap = dict(zip(tags["sample_id"], tags[col]))
        for slabel, sval in ((la, va), (lb, vb)):
            sel = [gid[s] for s in ids if tmap.get(s) == sval]
            if sel:
                by_slice[f"{sid}:{slabel}"] = np.mean(sel, axis=0)
    if by_slice:
        figures.gate_inspection(by_slice, modalities, out / "gate_inspection.png")
        logger.info("saved %s/gate_inspection.png", out)


def main(args):
    cfg = load_config(args.base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, level=args.log_level, filename="ablation.log")

    # 0. param accounting + matched-capacity invariant (independent of training)
    write_model_params(cfg, out)

    # 1. train (resume-aware). --co-train reads each batch once and trains all 4 systems
    # on it (one data pass per seed instead of one per system); otherwise train the
    # (system, seed) grid sequentially. Both write identical per-run outputs.
    if not args.skip_train:
        if args.co_train:
            run_cotrain(args)
        else:
            grid = [(s, sd) for s in SYSTEMS for sd in args.seeds]
            for system, seed in pbar(grid, desc="ablation grid", leave=True):
                run_training(system, seed, args)

    # 2. baselines
    baseline_metrics, bdir = run_baselines(args, out)

    # 3. collect trained results
    trained = {}
    for system in SYSTEMS:
        fms = []
        for seed in args.seeds:
            fp = Path(args.runs_dir) / f"{system}_seed{seed}" / "final_metrics.json"
            if fp.exists():
                fms.append(json.loads(fp.read_text()))
        trained[system] = fms

    # --- main table ---
    agg = aggregate(baseline_metrics, trained)
    (out / "main_table.md").write_text(render_main_markdown(agg))
    (out / "main_table.tex").write_text(render_main_latex(agg))
    print("\n" + render_main_markdown(agg))  # readable preview to stdout

    # --- stratified tables (uses first seed's predictions per system) ---
    # strata sources are precomputed columns in the timing parquet (S1/S2 timing,
    # S3 from text_context, S4/S5 from llm_*); no manifest/AMI access needed here.
    test_df = pd.read_parquet(Path(args.timing_dir) / "test.parquet")
    tags = tag_samples(test_df, pause_theta=cfg.get("eval", {}).get("pause_theta", 0.6))
    seed0 = args.seeds[0]
    preds_by_system = {}
    for system in SYSTEMS:
        rd = Path(args.runs_dir) / f"{system}_seed{seed0}"
        if (rd / "probs_test.npz").exists():
            preds_by_system[system] = load_probs(rd)
    strata = {}
    if preds_by_system:
        strata = stratified_metrics(tags, preds_by_system, min_cell=args.min_cell)
        (out / "stratified.md").write_text(
            f"_(stratified diagnostic from seed {seed0}; min cell size {args.min_cell})_\n\n"
            + render_stratified_markdown(strata, list(preds_by_system.keys())))
        with open(out / "stratified.json", "w") as f:
            json.dump(strata, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # --- figures ---
    # DET: trained (seed0) scores + rule curves from baseline CSVs
    trained_scores = {}
    for system, P in preds_by_system.items():
        trained_scores[DISPLAY[system]] = {"y_true": P["y_true"], "scores": 1.0 - P["probs"][:, 0]}
    rule_curves = {}
    det_eer = json.loads((bdir / "det_eer.json").read_text()) if (bdir / "det_eer.json").exists() else {}
    for rkey, rname in (("silence", "VA-Silence"), ("threshold", "VA-Threshold")):
        csv = bdir / f"det_curve_va_{rkey}.csv"
        if csv.exists() and rkey in det_eer:
            c = pd.read_csv(csv)
            rule_curves[rname] = {"false_alarm_rate": c["false_alarm_rate"].to_numpy(),
                                  "miss_rate": c["miss_rate"].to_numpy(), "eer": det_eer[rkey]["eer"]}
    try:
        figures.det_all_systems(trained_scores, rule_curves, out / "det_all_systems.png")
        logger.info("saved %s/det_all_systems.png", out)
    except Exception as e:  # noqa: BLE001
        logger.warning("DET figure skipped: %s", e)

    # confusion matrices (all rows)
    cms = {}
    for key, name in MAIN_ROWS:
        if key in TRAINED:
            fms = trained.get(key, [])
            if fms:
                cms[name] = np.array(fms[0]["test"]["confusion_matrix"])
        else:
            t = baseline_metrics.get(key, {}).get("test", {})
            if "confusion_matrix" in t:
                cms[name] = np.array(t["confusion_matrix"])
    if cms:
        figures.confusion_grid(cms, out / "confusion_matrices.png")
        logger.info("saved %s/confusion_matrices.png", out)

    # gate inspection (full system)
    try:
        gate_figure(args, cfg, args.runs_dir, tags, out)
    except Exception as e:  # noqa: BLE001
        logger.warning("gate figure skipped: %s", e)

    # --- bootstrap significance: full vs best single-modality (seed0 preds) ---
    bootstrap = None
    single = {k: agg[k]["macro_f1"][0] for k in ("timing", "audio_timing", "text_timing")
              if agg[k]["macro_f1"][0] is not None}
    if "full" in preds_by_system and single:
        best_single = max(single, key=single.get)
        if best_single in preds_by_system:
            from src.eval.metrics import bootstrap_macro_f1_diff
            P_full, P_bs = preds_by_system["full"], preds_by_system[best_single]
            boot_seed = args.seeds[0]
            bootstrap = bootstrap_macro_f1_diff(
                np.asarray(P_full["y_true"]), np.asarray(P_full["y_pred"]),
                np.asarray(P_bs["y_pred"]),
                reps=cfg.get("eval", {}).get("bootstrap_reps", 1000), seed=boot_seed,
                progress=True)
            bootstrap["comparison"] = f"full - {best_single}"
            bootstrap["seed"] = boot_seed
            logger.info("Bootstrap macro-F1 (full - %s, seed=%d, reps=%d): observed=%.3f "
                        "CI[%.3f, %.3f] frac(full>best single)=%.3f",
                        best_single, boot_seed, bootstrap["reps"], bootstrap["observed_diff"],
                        bootstrap["ci_low"], bootstrap["ci_high"], bootstrap["frac_gt_0"])

    # summary json
    summary = {"seeds": args.seeds, "systems": SYSTEMS, "bootstrap": bootstrap,
               "main_table": {k: {c: agg[k][c] for c, _h, _ in MAIN_COLS} for k, _ in MAIN_ROWS}}
    with open(out / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    logger.info("Ablation assembled under %s/ (main_table.md/.tex, model_params.{json,md}, "
                "stratified.md, *.png, ablation_summary.json)", out)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ami-root", default=None, help="Unused (kept for back-compat); strata read precomputed columns.")
    p.add_argument("--runs-dir", default="reports/runs")
    p.add_argument("--output", default="reports/ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    p.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=None,
                   help="Override train.max_epochs for every run.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-cell", type=int, default=50)
    p.add_argument("--amp", action="store_true", help="Pass --amp to training runs.")
    p.add_argument("--cache-format", default="auto", choices=["auto", "memmap", "per_file"],
                   help="Feature cache format passed to training: auto (packed memmap if "
                        "cache_packed/ is complete, else per-file), memmap, or per_file.")
    p.add_argument("--co-train", action="store_true",
                   help="Train all 4 systems per seed on ONE shared batch stream "
                        "(scripts/train_cotrain.py): ~4x less disk I/O, outputs verified "
                        "bit-identical to the sequential path. Otherwise train sequentially.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--skip-train", action="store_true")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
