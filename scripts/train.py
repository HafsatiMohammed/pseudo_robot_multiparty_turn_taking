#!/usr/bin/env python3
"""
Phase 5: train one system + one seed (config-driven, deterministic, resumable).

  - Loss: weighted cross-entropy (inverse-freq class weights with a mild BC
    up-weight) x per-sample training_weight.
  - AdamW + cosine schedule (optional linear warmup) + gradient clipping.
  - Early stopping on VALIDATION macro-F1 (mode=max): improvement iff
    metric > best + min_delta; stop after `patience` non-improving epochs.
  - Determinism: seeds (torch/cuda/numpy/python), deterministic algorithms,
    seeded workers, and a per-epoch reseed so data ordering + dropout are
    reproducible across resume.  (--amp opt-in; AMP may break bit-identity.)
  - Checkpointing: every epoch -> last.ckpt (atomic tmp->rename); on val-macroF1
    improvement -> best.ckpt. Each ckpt stores epoch, global_step, model/optim/sched
    state, best metric/epoch, patience counter, RNG states, full config, seed.
  - --resume <ckpt> restores ALL of the above and continues from epoch+1 so early
    stopping behaves identically to an uninterrupted run.
  - Saves softmax probabilities for val AND test (DET/EER + calibration).

Usage:
    python scripts/train.py --base configs/base.yaml --system timing --seed 42 \
        --timing-dir data/processed/timing --cache-dir data/processed/cache \
        --out reports/runs/timing_seed42 [--max-epochs N] [--max-samples M] \
        [--amp] [--warmup-epochs K] [--resume <ckpt>] [--log-level INFO]
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.multimodal import get_multimodal_dataloaders
from src.eval.metrics import compute_all, macro_f1, per_class_prf
from src.models.models_multimodal import SYSTEM_MODALITIES, build_system
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config, save_yaml
from src.utils.logging_setup import pbar, setup_logging
from src.utils.params import assert_frozen_zero, format_params_line, params_summary
from src.utils.seed import get_rng_states, seed_worker, set_rng_states, set_seed

logger = logging.getLogger(__name__)


def build_loss(train_ds, bc_upweight, device):
    cw = train_ds.get_class_weights().astype(np.float32)  # [3]
    cw[1] *= bc_upweight  # BACKCHANNEL index
    return torch.tensor(cw, device=device)


def run_epoch(model, loader, class_w, optimizer, device, grad_clip, train, *,
              scaler=None, use_amp=False, amp_device="cpu", desc="", log_every=50,
              sampler_sanity=False):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    torch.set_grad_enabled(train)
    bar = pbar(loader, desc=desc)
    last_norm = None
    for bi, batch in enumerate(bar):
        frame = batch["frame"].to(device); scalar = batch["scalar"].to(device)
        audio = batch["audio"].to(device); text = batch["text"].to(device)
        labels = batch["label"].to(device); sw = batch["weight"].to(device)

        if sampler_sanity and train and bi < 3:
            dist = Counter(batch["label"].tolist())
            logger.info("sampler batch %d label dist (0=WAIT,1=BC,2=START): %s", bi, dict(sorted(dist.items())))

        with torch.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(frame, scalar, audio, text)
            ce = F.cross_entropy(logits, labels, weight=class_w, reduction="none")
            loss = (ce * sw).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss.item()}) at {'train' if train else 'eval'} "
                               f"batch {bi}; stopping to avoid poisoning the run.")

        if train:
            optimizer.zero_grad()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    last_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    last_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss.item():.3f}",
                                lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if last_norm is not None and log_every and bi % log_every == 0:
                logger.debug("step %d grad_norm=%.3f", bi, float(last_norm))
        total_loss += loss.item() * len(labels); n += len(labels)
    torch.set_grad_enabled(True)
    return total_loss / max(n, 1), n


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    all_probs, all_true, all_ids = [], [], []
    for batch in pbar(loader, desc=desc):
        frame = batch["frame"].to(device); scalar = batch["scalar"].to(device)
        audio = batch["audio"].to(device); text = batch["text"].to(device)
        logits = model(frame, scalar, audio, text)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_true.append(batch["label"].numpy())
        all_ids.extend(batch["sample_id"])
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 3))
    y_true = np.concatenate(all_true) if all_true else np.zeros((0,), int)
    y_pred = probs.argmax(1) if len(probs) else np.zeros((0,), int)
    return y_true, y_pred, probs, all_ids


def save_probs(path, y_true, y_pred, probs, ids):
    np.savez(path, sample_id=np.array(ids), y_true=y_true, y_pred=y_pred, probs=probs)


def build_scheduler(optimizer, tcfg, max_epochs, warmup_epochs):
    sched_name = tcfg.get("scheduler", "cosine")
    if sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3), True
    if warmup_epochs and warmup_epochs > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warm = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
        return SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs]), False
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs), False


def prune_checkpoints(out_dir, keep_last_k):
    if keep_last_k <= 0:
        return
    epoch_ckpts = sorted(out_dir.glob("epoch_*.ckpt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    for p in epoch_ckpts[:-keep_last_k]:
        p.unlink(missing_ok=True)


def main(args):
    sys_path = _REPO_ROOT / "configs" / f"{args.system}.yaml" if args.system else None
    overrides = {"system": args.system} if args.system else {}
    max_epochs_cli = args.max_epochs if args.max_epochs is not None else args.epochs
    if max_epochs_cli is not None:
        overrides.setdefault("train", {})["max_epochs"] = max_epochs_cli
    cfg = load_config(args.base, sys_path if (sys_path and sys_path.exists()) else None, overrides)

    system = cfg["system"]
    seed = args.seed
    tcfg = cfg["train"]
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir, level=args.log_level)
    logger.info("system=%s seed=%d device=%s out=%s", system, seed, device, out_dir)

    set_seed(seed)
    logger.info("seed set (deterministic algorithms on; per-epoch reseed for resume identity)")

    # Modality gating: read ONLY the modalities this system uses (the model zeros the rest,
    # so gated zeros vs read-then-zeroed reals are identical). timing/scalar come from the
    # timing parquet, so only audio/text are gated here. Applies to the packed cache; on the
    # per-file cache it likewise skips reading .npy this system discards.
    read_modalities = set(SYSTEM_MODALITIES[system]) & {"audio", "text"}
    dls = get_multimodal_dataloaders(
        timing_dir=args.timing_dir, cache_dir=args.cache_dir,
        batch_size=tcfg["batch_size"], num_workers=args.num_workers,
        use_weighted_sampler=tcfg.get("use_weighted_sampler", True),
        max_samples=args.max_samples, seed=seed,
        worker_init_fn=seed_worker, generator=None,  # sampler uses global RNG (per-epoch reseed)
        cache_format=args.cache_format, read_modalities=read_modalities,
    )
    logger.info("cache_format=%s read_modalities=%s", args.cache_format, sorted(read_modalities))
    train_loader, val_loader, test_loader = dls["train"], dls["val"], dls["test"]
    train_ds = dls["train_dataset"]

    model = build_system(system, {"model": cfg["model"]}).to(device)
    assert_frozen_zero(model)  # encoders must be offline-cached, not in the graph
    psum = params_summary(model)
    logger.info(format_params_line(system, model))
    cfg["params"] = {k: psum[k] for k in ("total", "active", "trainable", "frozen")}

    class_w = build_loss(train_ds, tcfg.get("bc_upweight", 1.5), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01))
    max_epochs = tcfg["max_epochs"]
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else tcfg.get("warmup_epochs", 0)
    scheduler, plateau = build_scheduler(optimizer, tcfg, max_epochs, warmup_epochs)

    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = args.amp
    if use_amp:
        logger.warning("AMP enabled (--amp): may break bit-identical resume reproducibility.")
    scaler = torch.amp.GradScaler(amp_device, enabled=(use_amp and amp_device == "cuda")) \
        if use_amp and amp_device == "cuda" else None

    patience = tcfg.get("early_stop_patience", 10)
    min_delta = tcfg.get("min_delta", 1e-6)
    monitor = tcfg.get("monitor", "val_macro_f1")
    grad_clip = tcfg.get("grad_clip", 1.0)

    # ---- state (fresh or resumed) ----
    start_epoch = 0
    best_f1, best_epoch, no_improve, global_step = -1.0, -1, 0, 0
    if args.resume:
        ck = load_checkpoint(args.resume)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        best_f1, best_epoch = ck["best_val_macro_f1"], ck["best_epoch"]
        no_improve, global_step = ck["epochs_no_improve"], ck["global_step"]
        set_rng_states(ck.get("rng"))
        start_epoch = ck["epoch"] + 1
        logger.info("Resumed from %s: epoch %d -> start %d, best_f1=%.4f@%d, patience=%d/%d",
                    args.resume, ck["epoch"], start_epoch, best_f1, best_epoch, no_improve, patience)

    save_yaml(cfg, out_dir / "run_config.yaml")
    csv_path = out_dir / "metrics.csv"
    if not args.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"# system={system} seed={seed} total_params={psum['total']} "
                        f"active_params={psum['active']} frozen={psum['frozen']}"])
            w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1",
                        "val_wait_f1", "val_bc_f1", "val_start_f1", "lr", "epoch_sec", "samples_per_sec"])

    stop_reason = "reached max_epochs"
    t_train0 = time.perf_counter()
    epoch_bar = pbar(range(start_epoch, max_epochs), desc=f"{system} epochs", leave=True)
    for epoch in epoch_bar:
        epoch_seed = seed * 1_000_003 + epoch
        torch.manual_seed(epoch_seed); np.random.seed(epoch_seed % (2 ** 32))

        t0 = time.perf_counter()
        train_loss, n_train = run_epoch(
            model, train_loader, class_w, optimizer, device, grad_clip, train=True,
            scaler=scaler, use_amp=use_amp, amp_device=amp_device, desc=f"train e{epoch}",
            sampler_sanity=(epoch == start_epoch))
        epoch_sec = time.perf_counter() - t0
        sps = n_train / epoch_sec if epoch_sec > 0 else 0.0

        val_true, val_pred, _, _ = evaluate(model, val_loader, device, desc=f"val e{epoch}")
        val_loss, _ = run_epoch(model, val_loader, class_w, None, device, grad_clip, train=False,
                                use_amp=use_amp, amp_device=amp_device, desc=f"valloss e{epoch}")
        vf1 = macro_f1(val_true, val_pred)
        prf = per_class_prf(val_true, val_pred)
        global_step += len(train_loader)

        scheduler.step(vf1) if plateau else scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        improved = vf1 > best_f1 + min_delta
        if improved:
            best_f1, best_epoch, no_improve = vf1, epoch, 0
        else:
            no_improve += 1

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{vf1:.6f}",
                 f"{prf['WAIT']['f1']:.6f}", f"{prf['BACKCHANNEL']['f1']:.6f}",
                 f"{prf['START_SPEAKING']['f1']:.6f}", f"{lr_now:.6e}",
                 f"{epoch_sec:.2f}", f"{sps:.1f}"])

        # compact one-line summary (always printed; primary output under non-TTY)
        logger.info("epoch %d | train_loss %.4f | val_loss %.4f | val_macroF1 %.4f | "
                    "best %.4f @%d | patience %d/%d | lr %.2e | %.1f samp/s",
                    epoch, train_loss, val_loss, vf1, best_f1, best_epoch,
                    no_improve, patience, lr_now, sps)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(val_macroF1=f"{vf1:.3f}", best=f"{best_f1:.3f}")

        ckpt = {
            "epoch": epoch, "global_step": global_step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
            "epochs_no_improve": no_improve, "monitor": monitor,
            "rng": get_rng_states(), "config": cfg, "seed": seed, "system": system,
        }
        save_checkpoint(ckpt, out_dir / "last.ckpt")
        if improved:
            save_checkpoint(ckpt, out_dir / "best.ckpt")
        if args.keep_last_k > 0:
            save_checkpoint(ckpt, out_dir / f"epoch_{epoch}.ckpt")
            prune_checkpoints(out_dir, args.keep_last_k)

        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            logger.info("[debug] stop-after-epoch %d reached (simulated interruption); "
                        "scheduler horizon unchanged.", args.stop_after_epoch)
            stop_reason = "debug stop_after_epoch"
            break

        if no_improve >= patience:
            stop_reason = f"early stop (no {monitor} improvement in {patience} epochs)"
            logger.info(stop_reason + f" at epoch {epoch}")
            break

    total_train_sec = time.perf_counter() - t_train0
    logger.info("training done: %s | best %s=%.4f @ epoch %d | total %.1fs",
                stop_reason, monitor, best_f1, best_epoch, total_train_sec)

    # ---- final eval with BEST model; save probabilities for val + test ----
    best_path = out_dir / "best.ckpt"
    if best_path.exists():
        model.load_state_dict(load_checkpoint(best_path)["model"])
        logger.info("final checkpoint: %s (best %s=%.4f @ epoch %d)", best_path, monitor, best_f1, best_epoch)
    final = {"system": system, "seed": seed, "best_epoch": best_epoch,
             "best_val_macro_f1": best_f1, "stop_reason": stop_reason,
             "params": cfg["params"], "total_train_sec": total_train_sec}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        yt, yp, pr, ids = evaluate(model, loader, device, desc=f"final:{split}")
        save_probs(out_dir / f"probs_{split}.npz", yt, yp, pr, ids)
        final[split] = compute_all(yt, yp, pr)
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    logger.info("DONE %s seed=%d: test macroF1=%.4f FE=%s ME=%s EER=%s -> %s",
                system, seed, final["test"]["macro_f1"], final["test"]["false_entry"],
                final["test"]["missed_entry"], final["test"].get("eer"), out_dir)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--system", default=None, choices=["timing", "audio_timing", "text_timing", "full", None])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None, help="Override train.max_epochs.")
    p.add_argument("--epochs", type=int, default=None, help="Alias for --max-epochs (back-compat).")
    p.add_argument("--warmup-epochs", type=int, default=None, help="Linear warmup epochs (cosine only).")
    p.add_argument("--amp", action="store_true", help="Mixed precision (may break bit-identity).")
    p.add_argument("--keep-last-k", type=int, default=0, help="Also keep the last K epoch_*.ckpt.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache-format", default="auto", choices=["auto", "memmap", "per_file"],
                   help="auto: packed memmap if data/processed/cache_packed/ is complete, else "
                        "per-file; memmap: require packed; per_file: force per-file .npy.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="Debug/CI: break after this epoch (last.ckpt written) WITHOUT "
                        "changing the scheduler horizon -- used to test --resume.")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
