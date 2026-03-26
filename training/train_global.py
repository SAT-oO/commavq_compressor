#!/usr/bin/env python3
"""
Train NextFramePredictor on the commavq token dataset.

Final training profile used in docs:
    python training/train_global.py \\
        --shards 0 38 --val-shards 38 40 \\
        --epochs 40 --batch 192 --device auto \\
        --workers 16 --prefetch-factor 4

Resume after a crash:
    python training/train_global.py --auto-resume \\
        --shards 0 38 --epochs 40 --batch 192

Saves:
    resource/model.pt                       best model weights (float32)
    resource/model_f16.pt                   best model weights (float16, for submission)
    resource/global_freq.npy                marginal token frequency table
    resource/checkpoints/step_NNNNNNN.pt    rolling resume checkpoint (3 kept)
    resource/checkpoints/epoch_EEE_*.pt     end-of-epoch checkpoint (all kept)
    resource/checkpoints/best.pt            best-val-loss checkpoint (always kept)
"""

# ── Pin CPU thread counts before any library imports that spawn threads ──────
import os
_N_PHYSICAL = os.cpu_count() or 1   # logical on cloud; correct value set later
# These env-vars must be set before torch / numpy / OpenBLAS are imported.
os.environ.setdefault("OMP_NUM_THREADS",        str(_N_PHYSICAL))
os.environ.setdefault("MKL_NUM_THREADS",        str(_N_PHYSICAL))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   str(_N_PHYSICAL))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_N_PHYSICAL))
os.environ.setdefault("NUMEXPR_NUM_THREADS",    str(_N_PHYSICAL))
# Prevent HuggingFace tokenizers from spawning their own thread pool
# (we do our own multiprocessing in DataLoader).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import math
import multiprocessing
import random
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("torch not found. Install with the same Python you run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import (
    TOKENS_PER_FRAME,
    NextFramePredictor,
    build_context,
    save_model_f16,
)

# Hyper-parameters
PAIRS_PER_SAMPLE = 12    # random (context, target) pairs sampled per clip per epoch
LR               = 3e-4
WEIGHT_DECAY     = 1e-2
WARMUP_FRAC      = 0.05
GRAD_CLIP        = 1.0

# Checkpoint tiers (see docstring above)
CHECKPOINT_EVERY = 1000   # Tier-1: rolling, every N steps

MODEL_SAVE       = ROOT / "resource" / "model.pt"
MODEL_F16_SAVE   = ROOT / "resource" / "model_f16.pt"
GLOBAL_FREQ_SAVE = ROOT / "resource" / "global_freq.npy"
CHECKPOINT_DIR   = ROOT / "resource" / "checkpoints"
DATA_CACHE       = ROOT / "resource" / "dataset"


# Hardware setup

def configure_hardware(device: str, n_workers: int) -> tuple[bool, int]:
    """
    Apply device-specific performance settings.
    Returns (use_bf16, n_threads).
    """
    n_cpu = os.cpu_count() or 1

    # ── CPU threading (Xeon Platinum 8470 = 60 cores / 120 logical threads)
    # intra-op: parallelise a single operation across cores
    torch.set_num_threads(n_cpu)
    # inter-op: run independent operations concurrently (keep lower to avoid
    # cache thrashing; half of intra is a good default)
    torch.set_num_interop_threads(max(1, n_cpu // 2))

    use_bf16 = False

    if device == "cuda":
        # ── cuDNN: let it benchmark and pick fastest conv/matmul kernels
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.allow_tf32   = True   # faster FP32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        # ── TF32 precision for matmuls (free throughput on Ampere/Hopper)
        torch.set_float32_matmul_precision("high")

        # ── BF16: H100 has dedicated BF16 tensor cores (2× throughput vs FP32).
        #    Unlike FP16, BF16 keeps the same exponent range as FP32 so no
        #    gradient overflow / NaN risk → no GradScaler needed.
        use_bf16 = True

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}  ({vram_gb:.0f} GB VRAM)")
        print(f"  BF16 mixed precision: enabled")
        print(f"  cuDNN benchmark:      enabled")
        print(f"  TF32 matmul:          enabled")

    print(f"  CPU threads (intra-op): {torch.get_num_threads()}")
    print(f"  CPU threads (inter-op): {torch.get_num_interop_threads()}")
    print(f"  DataLoader workers:     {n_workers}")
    return use_bf16, n_cpu


# Checkpoint helpers

def _state(model):
    """Return state dict, unwrapping torch.compile wrapper if present."""
    raw = getattr(model, "_orig_mod", model)
    return raw.state_dict()


def _build_payload(model, optimizer, epoch, global_step,
                   best_val_loss, total_steps, warmup_steps, args_ns):
    return {
        "model":         _state(model),
        "optimizer":     optimizer.state_dict(),
        "epoch":         epoch,
        "global_step":   global_step,
        "best_val_loss": best_val_loss,
        "total_steps":   total_steps,
        "warmup_steps":  warmup_steps,
        "args":          vars(args_ns),
    }


def save_rolling_checkpoint(model, optimizer, epoch, global_step,
                             best_val_loss, total_steps, warmup_steps, args_ns):
    """Tier 1: rolling checkpoint every N steps; only 3 kept."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"step_{global_step:07d}.pt"
    torch.save(
        _build_payload(model, optimizer, epoch, global_step,
                       best_val_loss, total_steps, warmup_steps, args_ns),
        path,
    )
    for old in sorted(CHECKPOINT_DIR.glob("step_*.pt"))[:-3]:
        old.unlink(missing_ok=True)
    print(f"  [ckpt-rolling] {path.name}")


def save_epoch_checkpoint(model, optimizer, epoch, global_step,
                           best_val_loss, total_steps, warmup_steps, args_ns):
    """Tier 2: end-of-epoch checkpoint; never auto-deleted."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"epoch_{epoch:03d}_step_{global_step:07d}.pt"
    torch.save(
        _build_payload(model, optimizer, epoch, global_step,
                       best_val_loss, total_steps, warmup_steps, args_ns),
        path,
    )
    print(f"  [ckpt-epoch]   {path.name}")


def save_best_checkpoint(model, optimizer, epoch, global_step,
                          best_val_loss, total_steps, warmup_steps, args_ns):
    """Tier 3: best-val checkpoint; overwrites previous best."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "best.pt"
    raw  = getattr(model, "_orig_mod", model)
    torch.save(
        _build_payload(model, optimizer, epoch, global_step,
                       best_val_loss, total_steps, warmup_steps, args_ns),
        path,
    )
    torch.save(raw.state_dict(), MODEL_SAVE)
    save_model_f16(raw, MODEL_F16_SAVE)
    print(f"  [ckpt-best]    {path.name}  →  model.pt + model_f16.pt")


def load_checkpoint(path, model, optimizer, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    raw   = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {path}  (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    return ckpt


def latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    candidates = sorted(
        list(CHECKPOINT_DIR.glob("step_*.pt")) +
        list(CHECKPOINT_DIR.glob("epoch_*.pt"))
    )
    return candidates[-1] if candidates else None


# Dataset

class TokenDataset(Dataset):
    def __init__(self, tokens_list: list, pairs_per_sample: int = PAIRS_PER_SAMPLE):
        self.tokens = tokens_list
        self.pps    = pairs_per_sample

    def __len__(self) -> int:
        return len(self.tokens) * self.pps

    def __getitem__(self, idx: int):
        tokens = self.tokens[idx // self.pps]
        t      = random.randint(0, len(tokens) - 1)
        ctx    = build_context(tokens, t).astype(np.int64)
        target = tokens[t].astype(np.int64)
        return torch.from_numpy(ctx), torch.from_numpy(target)


# LR schedule: linear warmup → cosine decay

def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# Main

def main() -> None:
    n_cpu = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--shards", type=int, nargs=2, default=[0, 38],
                        metavar=("START", "END"),
                        help="Training shard range [START, END). Default: 0 38 (full dataset).")
    parser.add_argument("--val-shards", type=int, nargs=2, default=[38, 40],
                        metavar=("START", "END"),
                        help="Validation shard range. Default: 38 40.")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=512,
                        help="Per-GPU batch size. 512 saturates an H100 with this model. "
                             "Reduce if OOM.")
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--workers", "--num-workers", dest="workers", type=int, default=n_cpu,
                        help=f"DataLoader worker processes (alias: --num-workers). "
                             f"Defaults to all logical CPUs ({n_cpu}).")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Batches prefetched per DataLoader worker (default: 4).")
    parser.add_argument("--device",  default="auto",
                        help="'auto' selects cuda > mps > cpu.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (useful for debugging).")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable BF16 mixed precision even on CUDA.")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help=f"Rolling checkpoint interval in steps (default: {CHECKPOINT_EVERY}).")
    parser.add_argument("--resume-from", default=None, metavar="PATH",
                        help="Resume from a specific checkpoint file.")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-resume from the most recent checkpoint in resource/checkpoints/.")
    args = parser.parse_args()

    # ── Device selection ────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"Device: {device}  |  CPUs: {n_cpu}  |  Batch: {args.batch}")

    use_bf16, _ = configure_hardware(device, args.workers)
    if args.no_bf16:
        use_bf16 = False
    print(f"{'='*60}\n")

    # ── HuggingFace login (optional) ─────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass

    # ── Load dataset ─────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    train_shards = [f"data-{i:04d}.tar.gz" for i in range(*args.shards)]
    val_shards   = [f"data-{i:04d}.tar.gz" for i in range(*args.val_shards)]

    print(f"Loading train shards {args.shards[0]}–{args.shards[1]-1} …")
    ds_train = load_dataset(
        "commaai/commavq",
        num_proc=n_cpu,
        data_files={"train": train_shards},
        cache_dir=str(DATA_CACHE),
    )["train"]
    print(f"  {len(ds_train):,} training samples")

    print(f"Loading val shards {args.val_shards[0]}–{args.val_shards[1]-1} …")
    ds_val = load_dataset(
        "commaai/commavq",
        num_proc=n_cpu,
        data_files={"train": val_shards},
        cache_dir=str(DATA_CACHE),
    )["train"]
    print(f"  {len(ds_val):,} validation samples")

    # ── Pre-load tokens + compute global frequency table ─────────────────────
    print("\nPre-loading token arrays …")
    global_counts = np.zeros(1024, dtype=np.int64)

    def load_tokens(ds, desc):
        out = []
        for i, ex in enumerate(ds):
            t = np.array(ex["token.npy"], dtype=np.int16).reshape(1200, TOKENS_PER_FRAME)
            out.append(t)
            np.add.at(global_counts, t.ravel(), 1)
            if (i + 1) % 5000 == 0:
                print(f"  {desc}: {i+1:,}/{len(ds):,}", end="\r", flush=True)
        print()
        return out

    train_tokens = load_tokens(ds_train, "train")
    val_tokens   = load_tokens(ds_val,   "val  ")

    GLOBAL_FREQ_SAVE.parent.mkdir(parents=True, exist_ok=True)
    global_freq = global_counts.astype(np.float32) / global_counts.sum()
    np.save(GLOBAL_FREQ_SAVE, global_freq)
    marginal_h = -(global_freq * np.log2(global_freq + 1e-12)).sum()
    print(f"Global token entropy: {marginal_h:.2f} bits  "
          f"(marginal model gives {10/marginal_h:.2f}× compression)")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    pin = (device == "cuda")
    pf  = args.prefetch_factor if args.workers > 0 else None

    train_loader = DataLoader(
        TokenDataset(train_tokens, PAIRS_PER_SAMPLE),
        batch_size=args.batch, shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        prefetch_factor=pf,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        TokenDataset(val_tokens, PAIRS_PER_SAMPLE),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        prefetch_factor=pf,
        persistent_workers=(args.workers > 0),
    )
    print(f"Train batches/epoch: {len(train_loader):,}  "
          f"Val batches/epoch: {len(val_loader):,}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = NextFramePredictor().to(device)
    print(f"Model parameters: {model.param_count():,}  "
          f"(float16 ≈ {model.param_count()*2/1e6:.1f} MB)")

    # torch.compile: traces the model once and emits optimised CUDA kernels.
    # max-autotune benchmarks multiple kernel implementations and picks fastest.
    if not args.no_compile:
        try:
            print("Compiling model with torch.compile(mode='max-autotune') …")
            model = torch.compile(model, mode="max-autotune")
            print("  torch.compile OK")
        except Exception as e:
            print(f"  torch.compile skipped ({e})")

    optimizer = torch.optim.AdamW(
        (getattr(model, "_orig_mod", model)).parameters(),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
        fused=(device == "cuda"),   # fused AdamW kernel: faster on CUDA
    )
    criterion = nn.CrossEntropyLoss()

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")

    resume_path = Path(args.resume_from) if args.resume_from else None
    if args.auto_resume and not resume_path:
        resume_path = latest_checkpoint()
        print(f"--auto-resume: {'found ' + resume_path.name if resume_path else 'no checkpoint, starting fresh'}")

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch   = ckpt["epoch"]
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        total_steps   = ckpt.get("total_steps",  total_steps)
        warmup_steps  = ckpt.get("warmup_steps", warmup_steps)
        print(f"  → epoch {start_epoch}, step {global_step}, best_val {best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    # BF16 autocast: wraps forward pass so eligible ops run in bfloat16.
    # No GradScaler needed for BF16 (exponent range matches FP32).
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else
        torch.amp.autocast(device_type=device, enabled=False)
    )

    print(f"\nStarting training: epochs {start_epoch}–{args.epochs}, "
          f"{'BF16 ' if use_bf16 else ''}mixed precision\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for ctx, target in train_loader:
            # non_blocking=True overlaps H2D copy with previous GPU compute
            ctx    = ctx.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast_ctx:
                logits = model(ctx)                        # (B, S, V)
                loss   = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (getattr(model, "_orig_mod", model)).parameters(), GRAD_CLIP
            )
            optimizer.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % 200 == 0:
                avg = epoch_loss / n_batches
                print(f"  epoch {epoch}/{args.epochs}  step {global_step:,}  "
                      f"loss={avg:.4f}  ({2**avg:.2f} bits/tok)  lr={lr:.2e}")

            if global_step % args.checkpoint_every == 0:
                save_rolling_checkpoint(
                    model, optimizer, epoch, global_step, best_val_loss,
                    total_steps, warmup_steps, args,
                )

        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for ctx, target in val_loader:
                ctx    = ctx.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast_ctx:
                    logits = model(ctx)
                    val_loss += criterion(
                        logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
                    ).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        bpt       = val_loss / math.log(2)          # cross-entropy nats → bits
        est_ratio = 10.0 / bpt
        print(f"\nEpoch {epoch}/{args.epochs}  "
              f"train={train_loss/math.log(2):.3f} bits  "
              f"val={bpt:.3f} bits/tok  "
              f"→ ~{est_ratio:.2f}× compression\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_checkpoint(
                model, optimizer, epoch, global_step, best_val_loss,
                total_steps, warmup_steps, args,
            )

        save_epoch_checkpoint(
            model, optimizer, epoch + 1, global_step, best_val_loss,
            total_steps, warmup_steps, args,
        )

    best_bpt = best_val_loss / math.log(2)
    print(f"\nTraining complete.  "
          f"Best val: {best_bpt:.3f} bits/token  "
          f"→ ~{10/best_bpt:.2f}× compression")


if __name__ == "__main__":
    main()
