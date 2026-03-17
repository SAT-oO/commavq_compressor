#!/usr/bin/env python3
"""
Train NextFramePredictor on the commavq token dataset.

Usage:
    python training/train_global.py [--shards 0 4] [--epochs 5] [--batch 64]
    python training/train_global.py --resume-from resource/checkpoints/step_05000.pt

Saves:
    resource/model.pt                     best model (float32)
    resource/model_f16.pt                 best model (float16, used in submission)
    resource/global_freq.npy              marginal token frequency table
    resource/checkpoints/step_NNNNN.pt    periodic resume checkpoints
"""

import argparse
import math
import multiprocessing
import os
import random
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("torch not found. Install with the same Python you run:")
    print("  .venv/bin/python -m pip install -r requirements.txt")
    print("Then: .venv/bin/python training/train_global.py ...")
    sys.exit(1)

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import (
    CONTEXT_FRAMES,
    TOKENS_PER_FRAME,
    NextFramePredictor,
    build_context,
    save_model_f16,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
PAIRS_PER_SAMPLE  = 12    # random (context, target) windows sampled per sample
LR                = 3e-4
WEIGHT_DECAY      = 1e-2
WARMUP_FRAC       = 0.05  # fraction of total steps used for LR warmup
GRAD_CLIP         = 1.0
CHECKPOINT_EVERY  = 2000  # save a resume checkpoint every N steps

MODEL_SAVE      = ROOT / "resource" / "model.pt"
MODEL_F16_SAVE  = ROOT / "resource" / "model_f16.pt"
GLOBAL_FREQ_SAVE = ROOT / "resource" / "global_freq.npy"
CHECKPOINT_DIR  = ROOT / "resource" / "checkpoints"
DATA_CACHE      = ROOT / "resource" / "dataset"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, epoch, global_step, best_val_loss,
    total_steps, warmup_steps, args_ns,
):
    """Write a resumable checkpoint (model + optimizer + training state)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"step_{global_step:07d}.pt"
    torch.save(
        {
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "epoch":          epoch,
            "global_step":    global_step,
            "best_val_loss":  best_val_loss,
            "total_steps":    total_steps,
            "warmup_steps":   warmup_steps,
            "args":           vars(args_ns),
        },
        path,
    )
    # Keep only the 3 most recent checkpoints to save disk space.
    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    print(f"  ✓ checkpoint → {path.name}")
    return path


def load_checkpoint(path, model, optimizer, device):
    """Load a checkpoint and return the saved training state dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {path}  (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    return ckpt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Generates (context_frames, target_frame) pairs from a list of token arrays.

    tokens_list : list of (1200, 128) int16 numpy arrays
    """

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


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--shards", type=int, nargs=2, default=[0, 4],
                        metavar=("START", "END"),
                        help="Half-open range of shards to train on (default: 0 4). "
                             "Each shard is ~2500 samples, ~500 MB download. "
                             "Use 0 2 for a quick test, 0 38 for the full dataset.")
    parser.add_argument("--val-shards", type=int, nargs=2, default=[38, 40],
                        metavar=("START", "END"),
                        help="Shards held out for validation (default: 38 40)")
    parser.add_argument("--epochs",  type=int,   default=5)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--workers", type=int,   default=min(4, multiprocessing.cpu_count()))
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        metavar="N",
                        help=f"Save a resume checkpoint every N steps (default: {CHECKPOINT_EVERY})")
    parser.add_argument("--resume-from", default=None, metavar="CHECKPOINT_PATH",
                        help="Resume training from a checkpoint file, e.g. "
                             "resource/checkpoints/step_0005000.pt")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ---------------------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------------------
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (HuggingFace datasets library)")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
        except Exception:
            pass

    train_shards = [f"data-{i:04d}.tar.gz" for i in range(*args.shards)]
    val_shards   = [f"data-{i:04d}.tar.gz" for i in range(*args.val_shards)]

    print(f"Loading train shards {args.shards[0]}–{args.shards[1]-1} …")
    ds_train = load_dataset(
        "commaai/commavq",
        num_proc=multiprocessing.cpu_count(),
        data_files={"train": train_shards},
        cache_dir=str(DATA_CACHE),
    )["train"]
    print(f"  {len(ds_train):,} training samples")

    print(f"Loading val shards {args.val_shards[0]}–{args.val_shards[1]-1} …")
    ds_val = load_dataset(
        "commaai/commavq",
        num_proc=multiprocessing.cpu_count(),
        data_files={"train": val_shards},
        cache_dir=str(DATA_CACHE),
    )["train"]
    print(f"  {len(ds_val):,} validation samples")

    # ---------------------------------------------------------------------------
    # Preload token arrays + compute global token frequency table
    # ---------------------------------------------------------------------------
    print("Pre-loading token arrays and computing global frequency table …")
    global_counts = np.zeros(1024, dtype=np.int64)

    def load_tokens(ds, desc):
        out = []
        for i, ex in enumerate(ds):
            t = np.array(ex["token.npy"], dtype=np.int16).reshape(1200, TOKENS_PER_FRAME)
            out.append(t)
            np.add.at(global_counts, t.ravel(), 1)
            if (i + 1) % 2000 == 0:
                print(f"  {desc}: {i+1}/{len(ds)}", end="\r", flush=True)
        print()
        return out

    train_tokens = load_tokens(ds_train, "train")
    val_tokens   = load_tokens(ds_val,   "val  ")

    GLOBAL_FREQ_SAVE.parent.mkdir(parents=True, exist_ok=True)
    global_freq = global_counts.astype(np.float32) / global_counts.sum()
    np.save(GLOBAL_FREQ_SAVE, global_freq)
    marginal_entropy = -(global_freq * np.log2(global_freq + 1e-12)).sum()
    print(f"Global token entropy (marginal): {marginal_entropy:.2f} bits  →  "
          f"{10/marginal_entropy:.2f}× compression with marginal model")

    # ---------------------------------------------------------------------------
    # Datasets / loaders
    # ---------------------------------------------------------------------------
    train_ds = TokenDataset(train_tokens, PAIRS_PER_SAMPLE)
    val_ds   = TokenDataset(val_tokens,   PAIRS_PER_SAMPLE)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # ---------------------------------------------------------------------------
    # Model + optimiser
    # ---------------------------------------------------------------------------
    model = NextFramePredictor().to(device)
    print(f"Model parameters: {model.param_count():,}  "
          f"(float16 size ≈ {model.param_count()*2/1e6:.1f} MB)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))

    # ---------------------------------------------------------------------------
    # Optional: resume from checkpoint
    # ---------------------------------------------------------------------------
    start_epoch    = 1
    global_step    = 0
    best_val_loss  = float("inf")

    if args.resume_from:
        ckpt = load_checkpoint(args.resume_from, model, optimizer, device)
        start_epoch   = ckpt["epoch"]       # resume at the epoch that was in progress
        global_step   = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        # Respect the original LR schedule lengths if they differ.
        total_steps  = ckpt.get("total_steps",  total_steps)
        warmup_steps = ckpt.get("warmup_steps", warmup_steps)
        print(f"  Resuming: epoch {start_epoch}, step {global_step}, "
              f"best_val_loss {best_val_loss:.4f}")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for ctx, target in train_loader:
            ctx    = ctx.to(device)
            target = target.to(device)

            lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(ctx)
            loss   = criterion(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % 500 == 0:
                avg = epoch_loss / n_batches
                print(f"  epoch {epoch}/{args.epochs}  step {global_step}  "
                      f"train_loss={avg:.4f}  ({2**avg:.2f} bits/token)  lr={lr:.2e}")

            # Periodic checkpoint
            if global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step, best_val_loss,
                    total_steps, warmup_steps, args,
                )

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for ctx, target in val_loader:
                ctx    = ctx.to(device)
                target = target.to(device)
                logits = model(ctx)
                val_loss += criterion(
                    logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
                ).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        bits_per_token = 2 ** val_loss
        est_ratio      = 10.0 / bits_per_token
        print(f"Epoch {epoch}/{args.epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  ({bits_per_token:.2f} bits/token → "
              f"~{est_ratio:.2f}× compression)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            save_model_f16(model, MODEL_F16_SAVE)
            print(f"  ✓ best model → {MODEL_SAVE}")

        # End-of-epoch checkpoint (always save so you can resume between epochs)
        save_checkpoint(
            model, optimizer, epoch + 1, global_step, best_val_loss,
            total_steps, warmup_steps, args,
        )

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}  "
          f"({2**best_val_loss:.2f} bits/token → ~{10/2**best_val_loss:.2f}× compression)")


if __name__ == "__main__":
    main()
