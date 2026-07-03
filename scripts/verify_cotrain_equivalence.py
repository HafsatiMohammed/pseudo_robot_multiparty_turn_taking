#!/usr/bin/env python3
"""
Equivalence test: co-training MUST reproduce standalone per-system training.

Trains all four systems the standalone way (scripts/train.py, one process each) and the
co-trained way (scripts/train_cotrain.py, one shared batch stream) for the SAME seed /
epochs / data, then asserts that -- per system -- every per-epoch metric (train_loss,
val_loss, val_macro_f1, per-class F1, lr) and the final test macro-F1 match within a tight
tolerance. With determinism on and no --amp, they should be BIT-IDENTICAL (tol effectively 0).

This guards the paper's claim that the four ablation systems are trained the same way; the
only intended difference between the two paths is that co-training reads each batch once and
feeds it to all four models instead of re-reading it per system.

Usage (real data; mirrors the spec's smoke test):
    python scripts/verify_cotrain_equivalence.py --base configs/base.yaml \
        --timing-dir data/processed/timing --cache-dir data/processed/cache \
        --seed 42 --epochs 3 --max-samples 4000

Exit code 0 iff every system matches within --tol.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("timing", "audio_timing", "text_timing", "full")
# Numeric metrics.csv columns compared per epoch (skip epoch, timing columns).
COMPARE_COLS = ("train_loss", "val_loss", "val_macro_f1", "val_wait_f1", "val_bc_f1",
                "val_start_f1", "lr")


def _read_metrics(csv_path):
    """Parse metrics.csv (skip the leading '# system=...' comment row)."""
    import csv as _csv
    with open(csv_path) as f:
        rows = [r for r in _csv.reader(f) if r and not r[0].startswith("#")]
    header, data = rows[0], rows[1:]
    return header, [dict(zip(header, r)) for r in data]


def _run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def main(args):
    tmp = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="cotrain_eq_"))
    sa_dir = tmp / "standalone"
    co_dir = tmp / "cotrain"
    common = ["--base", args.base, "--timing-dir", args.timing_dir, "--cache-dir", args.cache_dir,
              "--device", args.device, "--num-workers", str(args.num_workers),
              "--max-epochs", str(args.epochs), "--log-level", "WARNING"]
    if args.max_samples is not None:
        common += ["--max-samples", str(args.max_samples)]

    print(f"[1/3] standalone: {len(SYSTEMS)} systems, seed {args.seed} -> {sa_dir}")
    for s in SYSTEMS:
        _run([sys.executable, "scripts/train.py", "--system", s, "--seed", str(args.seed),
              "--out", str(sa_dir / f"{s}_seed{args.seed}")] + common)

    print(f"[2/3] co-train: shared stream, seed {args.seed} -> {co_dir}")
    _run([sys.executable, "scripts/train_cotrain.py", "--seeds", str(args.seed),
          "--runs-dir", str(co_dir)] + common)

    print(f"[3/3] compare (tol={args.tol:g})")
    all_ok = True
    for s in SYSTEMS:
        sa_run, co_run = sa_dir / f"{s}_seed{args.seed}", co_dir / f"{s}_seed{args.seed}"
        _, sa_m = _read_metrics(sa_run / "metrics.csv")
        _, co_m = _read_metrics(co_run / "metrics.csv")
        sys_ok, worst, worst_col = True, 0.0, ""
        if len(sa_m) != len(co_m):
            sys_ok = False
            print(f"  [{s}] FAIL: epoch count differs (standalone {len(sa_m)} vs cotrain {len(co_m)})")
        else:
            for e, (ra, rc) in enumerate(zip(sa_m, co_m)):
                for c in COMPARE_COLS:
                    d = abs(float(ra[c]) - float(rc[c]))
                    if d > worst:
                        worst, worst_col = d, f"epoch{e}:{c}"
                    if d > args.tol:
                        sys_ok = False
        # final test macro-F1
        fa = json.loads((sa_run / "final_metrics.json").read_text())
        fc = json.loads((co_run / "final_metrics.json").read_text())
        df = abs(fa["test"]["macro_f1"] - fc["test"]["macro_f1"])
        if df > args.tol:
            sys_ok = False
        status = "PASS" if sys_ok else "FAIL"
        all_ok &= sys_ok
        print(f"  [{s}] {status} | worst per-epoch |diff|={worst:.2e} @ {worst_col} | "
              f"test macro-F1 |diff|={df:.2e}")

    print("\nRESULT:", "PASS -- co-training is equivalent to standalone" if all_ok
          else "FAIL -- co-training diverges from standalone (see above)")
    if not args.workdir:
        print(f"(runs kept in {tmp})")
    return 0 if all_ok else 1


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Max allowed |diff| per metric (bit-identical => ~0; spec allows <1e-4).")
    p.add_argument("--workdir", default=None, help="Where to put the two run trees (default: temp dir).")
    return p


if __name__ == "__main__":
    sys.exit(main(build_argparser().parse_args()))
