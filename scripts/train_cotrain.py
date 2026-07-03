#!/usr/bin/env python3
"""
Phase 5 (co-training): train ALL FOUR ablation systems on ONE shared batch stream.

Motivation
----------
The four systems (timing, audio_timing, text_timing, full) share IDENTICAL inputs
(timing + cached audio + text); they differ only in which modalities are active
(disabled ones are zeroed at their branch input by the model itself). Standalone
training reads the feature cache once PER SYSTEM (12 data passes for 4 systems x 3
seeds). Disk I/O -- many small random .npy reads -- is the bottleneck (GPU ~0%, the
net is ~1.6M params). This script reads each batch ONCE per seed and trains all four
models on it, turning 12 data passes into 3 (one per seed) => ~4x less disk I/O.

Equivalence (HARD requirement)
------------------------------
Each model must train EXACTLY as it would standalone with the same seed:
  * Same batches, same order, same weighted-sampler draw. We reproduce the standalone
    WeightedRandomSampler draw from the global RNG each epoch (identical multinomial),
    feed that fixed index order to the shared loader, and capture the post-draw RNG
    state S0 -- the exact state a standalone forward would see.
  * Same dropout stream per model. All four models forward the SAME batch inside one
    batch loop, so their dropout draws would otherwise interleave. We keep a per-model
    RNG snapshot: restore it before a model's step, save it after. Each model therefore
    sees the identical, uninterrupted dropout stream it would standalone (all four
    start every epoch from S0 -- reproduced here per model).
  * Independent everything else: each model has its own optimizer, scheduler, early-stop
    tracker, best/last checkpoints, metrics.csv, probs, final_metrics.json. No shared
    gradients or parameters -- only the input batch is shared.
  * Determinism/resume: the standalone per-epoch reseed (epoch_seed = seed*1_000_003 +
    epoch) is preserved, so every epoch is reproducible from epoch_seed alone and resume
    is bit-identical. Each model resumes from its own last.ckpt; models that already
    early-stopped are frozen (not updated/evaluated) while the rest keep training.

Outputs (identical layout to scripts/train.py, per system)
    reports/runs/<system>_seed<seed>/{last,best}.ckpt, metrics.csv,
    probs_{val,test}.npz, final_metrics.json, run_config.yaml

Usage
    python scripts/train_cotrain.py --base configs/base.yaml \
        --timing-dir data/processed/timing --cache-dir data/processed/cache \
        --runs-dir reports/runs --seeds 13 21 42 [--max-epochs N] [--max-samples M] \
        [--num-workers 12] [--device cuda] [--amp] [--resume]
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
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the standalone building blocks verbatim -- single source of truth, so the two
# paths cannot drift (loss weighting, scheduler, checkpoint pruning, probs I/O).
from scripts.train import build_loss, build_scheduler, prune_checkpoints, save_probs
from src.data.multimodal import MultimodalDataset, collate_multimodal, resolve_packed_dir
from src.eval.metrics import compute_all, macro_f1, per_class_prf
from src.models.models_multimodal import SYSTEM_MODALITIES, build_system
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config, save_yaml
from src.utils.logging_setup import pbar, setup_logging
from src.utils.params import assert_frozen_zero, format_params_line, params_summary
from src.utils.seed import get_rng_states, seed_worker, set_rng_states, set_seed

logger = logging.getLogger(__name__)

SYSTEMS = ("timing", "audio_timing", "text_timing", "full")


# ---------------------------------------------------------------------------
# Shared batch stream: reproduce the standalone sampler draw, feed a fixed order
# ---------------------------------------------------------------------------
class _FixedOrderSampler(Sampler):
    """Yields a pre-computed index order set per epoch (main-process, so it works with
    persistent workers). The order is the exact draw the standalone WeightedRandomSampler
    would produce for the epoch's RNG state."""

    def __init__(self, n: int):
        self._indices = list(range(n))

    def set_indices(self, indices) -> None:
        self._indices = list(indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def _build_datasets(timing_dir, cache_dir, max_samples, require_cache=True, cache_format="auto"):
    """Train/val/test datasets with val/test reusing TRAIN normalization stats
    (identical to get_multimodal_dataloaders). Co-training reads ALL modalities (every system
    trains on the shared batch), so no modality gating -- only the packed-cache I/O win applies."""
    resolved = resolve_packed_dir(cache_dir, cache_format, None)
    if resolved is not None:
        logger.info("co-train cache: PACKED memmap at %s", resolved)
    pk = str(resolved) if resolved else None
    timing_dir = Path(timing_dir)
    train_ds = MultimodalDataset(parquet_path=str(timing_dir / "train.parquet"),
                                 cache_dir=cache_dir, split="train", normalize=True,
                                 require_cache=require_cache, packed_dir=pk).truncate(max_samples)
    stats = train_ds.norm_stats()
    val_ds = MultimodalDataset(parquet_path=str(timing_dir / "validation.parquet"),
                               cache_dir=cache_dir, split="validation", normalize=True,
                               require_cache=require_cache, norm_stats=stats, packed_dir=pk).truncate(max_samples)
    test_ds = MultimodalDataset(parquet_path=str(timing_dir / "test.parquet"),
                                cache_dir=cache_dir, split="test", normalize=True,
                                require_cache=require_cache, norm_stats=stats, packed_dir=pk).truncate(max_samples)
    return train_ds, val_ds, test_ds


def _loader(dataset, batch_size, num_workers, *, sampler=None, shuffle=False,
            drop_last=False, pin_memory=False, generator=None):
    # A FIXED `generator` is passed so the DataLoader draws its per-iterator base_seed
    # (torch.empty(()).random_(generator=...)) from THAT generator, NOT the global RNG.
    # The global RNG then reflects only what a standalone epoch consumes (base_seed +
    # sampler multinomial), which we reproduce by hand in the epoch loop -- see train_seed.
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              collate_fn=collate_multimodal, worker_init_fn=seed_worker,
              pin_memory=pin_memory, generator=generator)
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    if sampler is not None:
        return DataLoader(dataset, sampler=sampler, drop_last=drop_last, **kw)
    return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, **kw)


# ---------------------------------------------------------------------------
# Combined evaluation: read val/test ONCE, run every (active) model on the batch
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_multi(models, loader, class_w, device, *, use_amp=False, amp_device="cpu",
                   desc="eval"):
    """One pass over `loader`, evaluating all models in `models` (dict name->model).
    Returns {name: (y_true, y_pred, probs, ids, loss)}. Reproduces standalone's
    evaluate() (softmax/argmax) AND run_epoch(train=False) weighted-CE val loss, but
    with a single read of the split shared across models."""
    for m in models.values():
        m.eval()
    acc = {n: {"probs": [], "loss": 0.0} for n in models}
    y_true, ids, n = [], [], 0
    for batch in pbar(loader, desc=desc):
        frame = batch["frame"].to(device); scalar = batch["scalar"].to(device)
        audio = batch["audio"].to(device); text = batch["text"].to(device)
        labels = batch["label"].to(device); sw = batch["weight"].to(device)
        y_true.append(batch["label"].numpy()); ids.extend(batch["sample_id"])
        n += len(labels)
        for name, model in models.items():
            with torch.autocast(device_type=amp_device, enabled=use_amp):
                logits = model(frame, scalar, audio, text)
                ce = F.cross_entropy(logits, labels, weight=class_w, reduction="none")
                loss = (ce * sw).mean()
            acc[name]["probs"].append(torch.softmax(logits, dim=1).float().cpu().numpy())
            acc[name]["loss"] += loss.item() * len(labels)
    y_true = np.concatenate(y_true) if y_true else np.zeros((0,), int)
    out = {}
    for name in models:
        probs = np.concatenate(acc[name]["probs"]) if acc[name]["probs"] else np.zeros((0, 3))
        y_pred = probs.argmax(1) if len(probs) else np.zeros((0,), int)
        out[name] = (y_true, y_pred, probs, list(ids), acc[name]["loss"] / max(n, 1))
    return out


# ---------------------------------------------------------------------------
# Per-model training state (mirrors the locals in scripts.train.main)
# ---------------------------------------------------------------------------
class _ModelState:
    def __init__(self, name, model, optimizer, scheduler, plateau, cfg, out_dir):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.plateau = plateau
        self.cfg = cfg
        self.out_dir = out_dir
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.no_improve = 0
        self.global_step = 0
        self.stopped = False
        self.stop_reason = "reached max_epochs"
        self.next_epoch = 0          # first epoch this model still needs to run
        self.train_loss = 0.0        # per-epoch accumulator
        self.n_seen = 0
        self.rng = None              # per-model dropout RNG snapshot within an epoch


def _seed_state_for_init(seed):
    """Reproduce the exact RNG state a standalone run is in right before it builds its
    model: get_multimodal_dataloaders() does torch.manual_seed(seed)+np.random.seed(seed)
    and consumes NO RNG before build_system()."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_seed(args, seed):
    """Co-train all four systems for one seed on a shared batch stream."""
    base_cfg = load_config(args.base)
    tcfg = base_cfg["train"]
    device = args.device
    runs_dir = Path(args.runs_dir)

    max_epochs = args.max_epochs if args.max_epochs is not None else tcfg["max_epochs"]
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else tcfg.get("warmup_epochs", 0)
    patience = tcfg.get("early_stop_patience", 10)
    min_delta = tcfg.get("min_delta", 1e-6)
    monitor = tcfg.get("monitor", "val_macro_f1")
    grad_clip = tcfg.get("grad_clip", 1.0)
    use_weighted_sampler = tcfg.get("use_weighted_sampler", True)
    batch_size = tcfg["batch_size"]
    val_bs = batch_size * 2

    out_dirs = {name: runs_dir / f"{name}_seed{seed}" for name in SYSTEMS}

    # ---- skip if every system for this seed is already complete ----
    if all((out_dirs[n] / "final_metrics.json").exists() and (out_dirs[n] / "probs_test.npz").exists()
           for n in SYSTEMS):
        logger.info("[skip] seed%d (all 4 systems complete)", seed)
        return

    set_seed(seed)
    logger.info("co-train seed=%d device=%s max_epochs=%d (deterministic; per-epoch reseed)",
                seed, device, max_epochs)

    # ---- datasets (val/test reuse train norm stats), built once; no RNG consumed ----
    train_ds, val_ds, test_ds = _build_datasets(args.timing_dir, args.cache_dir, args.max_samples,
                                                cache_format=args.cache_format)

    pin = str(device).startswith("cuda")
    drop_last = len(train_ds) > batch_size
    # Fixed generator isolates every loader's base_seed from the global RNG (see _loader).
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    train_sampler = _FixedOrderSampler(len(train_ds))
    train_loader = _loader(train_ds, batch_size, args.num_workers, sampler=train_sampler,
                           drop_last=drop_last, pin_memory=pin, generator=loader_gen)
    val_loader = _loader(val_ds, val_bs, args.num_workers, shuffle=False, pin_memory=pin,
                         generator=loader_gen)
    test_loader = _loader(test_ds, val_bs, args.num_workers, shuffle=False, pin_memory=pin,
                          generator=loader_gen)

    # Reference sampler used ONLY to reproduce the standalone draw from the global RNG.
    # It draws indices (consuming the global RNG exactly as standalone) but loads no data.
    if use_weighted_sampler:
        cw = train_ds.get_class_weights()
        sample_w = cw[train_ds.labels]
        ref_sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(train_ds),
                                            replacement=True, generator=None)
    else:
        ref_sampler = RandomSampler(train_ds, generator=None)

    class_w = build_loss(train_ds, tcfg.get("bc_upweight", 1.5), device)

    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = args.amp
    if use_amp:
        logger.warning("AMP enabled (--amp): may break bit-identical equivalence/resume.")
    scalers = {n: (torch.amp.GradScaler(amp_device, enabled=True)
                   if (use_amp and amp_device == "cuda") else None) for n in SYSTEMS}

    # ---- build the four models (each initialized from the SAME RNG state standalone
    # would use -> identical initial parameters) + per-model optimizer/scheduler ----
    states = {}
    for name in SYSTEMS:
        _seed_state_for_init(seed)                        # reproduce standalone init RNG
        model = build_system(name, {"model": base_cfg["model"]}).to(device)
        assert_frozen_zero(model)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                weight_decay=tcfg.get("weight_decay", 0.01))
        sched, plateau = build_scheduler(opt, tcfg, max_epochs, warmup_epochs)
        cfg = load_config(args.base, overrides={"system": name,
                                                "train": {"max_epochs": max_epochs}})
        psum = params_summary(model)
        cfg["params"] = {k: psum[k] for k in ("total", "active", "trainable", "frozen")}
        states[name] = _ModelState(name, model, opt, sched, plateau, cfg, out_dirs[name])
        logger.info(format_params_line(name, model))

    # matched-capacity invariant (per model, all equal)
    totals = {n: params_summary(s.model)["total"] for n, s in states.items()}
    assert len(set(totals.values())) == 1, f"Matched-capacity violated: {totals}"
    logger.info("Matched capacity OK: all systems = %d params", next(iter(totals.values())))

    # ---- resume each model independently from its own last.ckpt ----
    for name, st in states.items():
        last = st.out_dir / "last.ckpt"
        if args.resume and last.exists():
            ck = load_checkpoint(last)
            st.model.load_state_dict(ck["model"])
            st.optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                st.scheduler.load_state_dict(ck["scheduler"])
            st.best_f1, st.best_epoch = ck["best_val_macro_f1"], ck["best_epoch"]
            st.no_improve, st.global_step = ck["epochs_no_improve"], ck["global_step"]
            st.next_epoch = ck["epoch"] + 1
            if st.no_improve >= patience or st.next_epoch >= max_epochs:
                st.stopped = True
                st.stop_reason = (f"early stop (no {monitor} improvement in {patience} epochs)"
                                  if st.no_improve >= patience else "reached max_epochs")
            logger.info("[resume] %s seed%d: -> epoch %d, best_f1=%.4f@%d, patience=%d/%d%s",
                        name, seed, st.next_epoch, st.best_f1, st.best_epoch, st.no_improve,
                        patience, " (stopped)" if st.stopped else "")

    # persist per-system run_config.yaml + (re)initialize metrics.csv
    for name, st in states.items():
        st.out_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(st.cfg, st.out_dir / "run_config.yaml")
        csv_path = st.out_dir / "metrics.csv"
        if not (args.resume and csv_path.exists()):
            psum = st.cfg["params"]
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([f"# system={name} seed={seed} total_params={psum['total']} "
                            f"active_params={psum['active']} frozen={psum['frozen']}"])
                w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1",
                            "val_wait_f1", "val_bc_f1", "val_start_f1", "lr", "epoch_sec",
                            "samples_per_sec"])

    # The shared loop resumes at the max epoch any *trainable* model still needs.
    start_epoch = max([st.next_epoch for st in states.values() if not st.stopped], default=max_epochs)
    for name, st in states.items():
        if not st.stopped and st.next_epoch < start_epoch:
            logger.warning("%s behind (epoch %d < %d); resuming it late on the shared stream.",
                           name, st.next_epoch, start_epoch)

    t_train0 = time.perf_counter()
    epoch_bar = pbar(range(start_epoch, max_epochs), desc=f"cotrain seed{seed} epochs", leave=True)
    for epoch in epoch_bar:
        active = [n for n in SYSTEMS if not states[n].stopped]
        if not active:
            logger.info("all systems stopped; ending co-train loop at epoch %d", epoch)
            break

        # --- reproduce the standalone epoch RNG consumption, capture S0 ---
        # A standalone epoch consumes the global RNG in this exact order before the first
        # forward: (1) the DataLoader iterator's base_seed draw, (2) the WeightedRandomSampler
        # multinomial. We reproduce BOTH so S0 -- the state the first forward's dropout sees --
        # and the drawn index order are bit-identical to standalone.
        epoch_seed = seed * 1_000_003 + epoch
        torch.manual_seed(epoch_seed); np.random.seed(epoch_seed % (2 ** 32))
        torch.empty((), dtype=torch.int64).random_()   # (1) standalone DataLoader base_seed
        indices = list(iter(ref_sampler))              # (2) identical multinomial draw
        train_sampler.set_indices(indices)
        rng_S0 = get_rng_states()                  # state each model's forward starts from
        for n in active:
            states[n].rng = rng_S0                 # read-only shared start; diverges after use
            states[n].train_loss = 0.0; states[n].n_seen = 0
            states[n].model.train()

        torch.set_grad_enabled(True)
        t0 = time.perf_counter()
        for bi, batch in enumerate(pbar(train_loader, desc=f"train e{epoch}")):
            frame = batch["frame"].to(device); scalar = batch["scalar"].to(device)
            audio = batch["audio"].to(device); text = batch["text"].to(device)
            labels = batch["label"].to(device); sw = batch["weight"].to(device)
            if epoch == start_epoch and bi < 3:
                dist = Counter(batch["label"].tolist())
                logger.info("shared sampler batch %d label dist (0=WAIT,1=BC,2=START): %s",
                            bi, dict(sorted(dist.items())))
            for name in active:
                st = states[name]
                set_rng_states(st.rng)             # restore this model's dropout stream
                with torch.autocast(device_type=amp_device, enabled=use_amp):
                    logits = st.model(frame, scalar, audio, text)
                    ce = F.cross_entropy(logits, labels, weight=class_w, reduction="none")
                    loss = (ce * sw).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss ({loss.item()}) for {name} at batch "
                                       f"{bi}, epoch {epoch}; stopping to avoid poisoning.")
                st.optimizer.zero_grad()
                sc = scalers[name]
                if sc is not None and sc.is_enabled():
                    sc.scale(loss).backward()
                    if grad_clip:
                        sc.unscale_(st.optimizer)
                        torch.nn.utils.clip_grad_norm_(st.model.parameters(), grad_clip)
                    sc.step(st.optimizer); sc.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(st.model.parameters(), grad_clip)
                    st.optimizer.step()
                st.rng = get_rng_states()           # save advanced dropout stream
                st.train_loss += loss.item() * len(labels); st.n_seen += len(labels)
        epoch_sec = time.perf_counter() - t0

        # --- combined val pass (one read, all active models) ---
        val_out = evaluate_multi({n: states[n].model for n in active}, val_loader, class_w,
                                 device, use_amp=use_amp, amp_device=amp_device, desc=f"val e{epoch}")

        n_active = len(active)
        for name in active:
            st = states[name]
            yt, yp, _, _, val_loss = val_out[name]
            vf1 = macro_f1(yt, yp)
            prf = per_class_prf(yt, yp)
            st.global_step += len(train_loader)
            st.scheduler.step(vf1) if st.plateau else st.scheduler.step()
            lr_now = st.optimizer.param_groups[0]["lr"]
            train_loss = st.train_loss / max(st.n_seen, 1)
            # per-model view of throughput (epoch shared across n_active models)
            sps = (st.n_seen / epoch_sec) if epoch_sec > 0 else 0.0

            improved = vf1 > st.best_f1 + min_delta
            if improved:
                st.best_f1, st.best_epoch, st.no_improve = vf1, epoch, 0
            else:
                st.no_improve += 1

            with open(st.out_dir / "metrics.csv", "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{vf1:.6f}",
                     f"{prf['WAIT']['f1']:.6f}", f"{prf['BACKCHANNEL']['f1']:.6f}",
                     f"{prf['START_SPEAKING']['f1']:.6f}", f"{lr_now:.6e}",
                     f"{epoch_sec:.2f}", f"{sps:.1f}"])
            logger.info("[%s] epoch %d | train_loss %.4f | val_loss %.4f | val_macroF1 %.4f | "
                        "best %.4f @%d | patience %d/%d | lr %.2e",
                        name, epoch, train_loss, val_loss, vf1, st.best_f1, st.best_epoch,
                        st.no_improve, patience, lr_now)

            ckpt = {
                "epoch": epoch, "global_step": st.global_step,
                "model": st.model.state_dict(), "optimizer": st.optimizer.state_dict(),
                "scheduler": st.scheduler.state_dict(),
                "best_val_macro_f1": st.best_f1, "best_epoch": st.best_epoch,
                "epochs_no_improve": st.no_improve, "monitor": monitor,
                "rng": get_rng_states(), "config": st.cfg, "seed": seed, "system": name,
            }
            save_checkpoint(ckpt, st.out_dir / "last.ckpt")
            if improved:
                save_checkpoint(ckpt, st.out_dir / "best.ckpt")
            if args.keep_last_k > 0:
                save_checkpoint(ckpt, st.out_dir / f"epoch_{epoch}.ckpt")
                prune_checkpoints(st.out_dir, args.keep_last_k)

            if st.no_improve >= patience:
                st.stopped = True
                st.stop_reason = f"early stop (no {monitor} improvement in {patience} epochs)"
                logger.info("[%s] %s at epoch %d", name, st.stop_reason, epoch)

        agg_sps = (sum(states[n].n_seen for n in active) / epoch_sec) if epoch_sec > 0 else 0.0
        logger.info("epoch %d done | %.1fs | %d active model(s) | aggregate %.1f samp/s "
                    "(one read served all)", epoch, epoch_sec, n_active, agg_sps)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(active=n_active,
                                  best_full=f"{states['full'].best_f1:.3f}")

        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            # Debug/CI only: simulate an interruption. last.ckpt for every model is on
            # disk; leave NO final_metrics so --resume picks the seed back up. The
            # scheduler horizon is unchanged, so resume is bit-identical.
            logger.info("[debug] stop-after-epoch %d reached (simulated interruption); "
                        "skipping final eval so --resume continues this seed.",
                        args.stop_after_epoch)
            return

    total_train_sec = time.perf_counter() - t_train0

    # ---- final eval with each model's BEST weights; shared val + test reads ----
    for name, st in states.items():
        best_path = st.out_dir / "best.ckpt"
        if best_path.exists():
            st.model.load_state_dict(load_checkpoint(best_path)["model"])
            logger.info("[%s] final checkpoint: %s (best %s=%.4f @ epoch %d)",
                        name, best_path, monitor, st.best_f1, st.best_epoch)
    all_models = {n: states[n].model for n in SYSTEMS}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        res = evaluate_multi(all_models, loader, class_w, device, use_amp=use_amp,
                             amp_device=amp_device, desc=f"final:{split}")
        for name, st in states.items():
            yt, yp, pr, ids, _ = res[name]
            save_probs(st.out_dir / f"probs_{split}.npz", yt, yp, pr, ids)
            if not hasattr(st, "final"):
                st.final = {"system": name, "seed": seed, "best_epoch": st.best_epoch,
                            "best_val_macro_f1": st.best_f1, "stop_reason": st.stop_reason,
                            "params": st.cfg["params"], "total_train_sec": total_train_sec}
            st.final[split] = compute_all(yt, yp, pr)

    for name, st in states.items():
        with open(st.out_dir / "final_metrics.json", "w") as f:
            json.dump(st.final, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, np.floating) else o)
        logger.info("DONE %s seed=%d: test macroF1=%.4f FE=%s ME=%s EER=%s -> %s",
                    name, seed, st.final["test"]["macro_f1"], st.final["test"]["false_entry"],
                    st.final["test"]["missed_entry"], st.final["test"].get("eer"), st.out_dir)
    logger.info("co-train seed=%d complete: %s | total %.1fs",
                seed, {n: f"{states[n].best_f1:.4f}@{states[n].best_epoch}" for n in SYSTEMS},
                total_train_sec)


def main(args):
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(runs_dir, level=args.log_level, filename="cotrain.log")
    for seed in args.seeds:
        train_seed(args, seed)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--runs-dir", default="reports/runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-epochs", "--epochs", dest="max_epochs", type=int, default=None,
                   help="Override train.max_epochs for every system.")
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Mixed precision (breaks bit-identity).")
    p.add_argument("--keep-last-k", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache-format", default="auto", choices=["auto", "memmap", "per_file"],
                   help="auto: packed memmap if cache_packed/ is complete, else per-file.")
    p.add_argument("--resume", action="store_true",
                   help="Resume each system from its own last.ckpt (skips fully-complete seeds).")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="Debug/CI: simulate an interruption after this epoch (last.ckpt "
                        "written, no final_metrics) -- used to test --resume.")
    p.add_argument("--log-level", default="INFO")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
