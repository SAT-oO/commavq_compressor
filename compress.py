#!/usr/bin/env python3
"""
Compress commavq token sequences using a trained NextFramePredictor model.

Usage:
    python compress.py [--model resource/model.pt] [--output submission.zip]
                       [--splits data-0000.tar.gz data-0001.tar.gz]

The output zip contains:
    decompress.py            – decompression entry-point
    training/model.py        – model architecture
    training/coder.py        – range-coding wrappers
    model_weights.pt       – float16 model weights
    global_freq.npy        – marginal token frequency table
    compressed_data.pkl    – compressed bitstreams keyed by file_name
"""

import argparse
import io
import multiprocessing
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from training.coder import FrameEncoder
from training.model import (
    CONTEXT_FRAMES,
    TOKENS_PER_FRAME,
    NextFramePredictor,
    build_context_batch,
    load_model,
    rebuild_from_f16,
    save_model_f16,
)

ENCODE_BATCH   = 128    # samples grouped per outer batch
FRAME_CHUNK    = 64     # time-step chunks within one sample's inference


# Core compression routine

def _predict_all_frames(
    tokens: np.ndarray,             # (1200, 128) int16 for one sample
    model: NextFramePredictor,
    device: str,
    global_tile: np.ndarray,        # (128, 1024) float32
) -> np.ndarray:
    """
    Run model once per sample, batching all 1200 time steps at once.
    Returns (1200, 128, 1024) float32 probability arrays.
    """
    T_len = len(tokens)
    batch = tokens[None]    # (1, T_len, 128)

    # Build all context windows: (T_len, T, 128)
    all_ctx = np.stack([build_context_batch(batch, t)[0] for t in range(T_len)])

    all_probs = []
    with torch.no_grad():
        for start in range(0, T_len, FRAME_CHUNK):
            end = min(start + FRAME_CHUNK, T_len)
            x = torch.tensor(all_ctx[start:end].astype(np.int64), device=device)
            logits = model(x)                               # (chunk, 128, 1024)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs_all = np.concatenate(all_probs, axis=0)          # (T_len, 128, 1024)
    probs_all[0] = global_tile                              # override frame 0
    return probs_all


def compress_batch(
    batch_tokens: np.ndarray,       # (B, 1200, 128) int16
    model: NextFramePredictor,
    global_probs: np.ndarray,       # (1024,) float32
    device: str,
) -> List[bytes]:
    """
    Compress B token sequences.  For each sample we run the model across
    all 1200 time steps in one batched inference call, then entropy-code.
    """
    B, num_frames, _ = batch_tokens.shape

    global_tile = np.broadcast_to(
        global_probs[None, :], (TOKENS_PER_FRAME, len(global_probs))
    ).astype(np.float32)

    results = []
    for i in range(B):
        probs = _predict_all_frames(batch_tokens[i], model, device, global_tile)
        enc = FrameEncoder()
        for t in range(num_frames):
            enc.encode_frame(batch_tokens[i, t], probs[t])
        results.append(enc.to_bytes())

    return results


# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="resource/model.pt")
    parser.add_argument("--global-freq", default="resource/global_freq.npy")
    parser.add_argument("--data-cache",  default="resource/dataset")
    parser.add_argument("--output",      default="submission.zip")
    parser.add_argument("--splits", nargs="+",
                        default=["data-0000.tar.gz", "data-0001.tar.gz"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Device selection
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

    # Load model
    print(f"Loading model from {args.model} …")
    model_orig = load_model(args.model, device=device)
    print(f"  {model_orig.param_count():,} parameters")

    # Round-trip through float16 so that the encoder uses IDENTICALLY quantised
    # weights as decompress.py will load from the zip.
    f16_buf = io.BytesIO()
    save_model_f16(model_orig, f16_buf)
    f16_buf.seek(0)
    model = rebuild_from_f16(f16_buf, device=device)

    # Compile for faster inference (PyTorch ≥ 2.0, skipped if unavailable).
    try:
        model = torch.compile(model)
        print("  torch.compile() applied")
    except Exception:
        pass

    # Load global frequency table
    global_probs = np.load(args.global_freq).astype(np.float32)
    global_probs = np.clip(global_probs, 1e-6, None)
    global_probs /= global_probs.sum()

    # Load dataset
    print(f"Loading dataset splits: {args.splits} …")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    ds = load_dataset(
        "commaai/commavq",
        num_proc=multiprocessing.cpu_count(),
        data_files={"train": args.splits},
        cache_dir=args.data_cache,
    )["train"]
    print(f"  {len(ds):,} samples")

    # Load all examples into memory (needed for batch-parallel processing).
    print("Loading token arrays …")
    examples = list(ds)   # triggers Arrow cache read, fast on subsequent runs

    compressed_index: Dict[str, bytes] = {}
    total_raw_bytes = 0

    print("Compressing …")
    for b_start in range(0, len(examples), ENCODE_BATCH):
        b_end  = min(b_start + ENCODE_BATCH, len(examples))
        batch  = examples[b_start:b_end]

        file_names = [ex["json"]["file_name"] for ex in batch]
        batch_tokens = np.stack(
            [np.array(ex["token.npy"], dtype=np.int16).reshape(1200, TOKENS_PER_FRAME)
             for ex in batch]
        )                                                           # (B, 1200, 128)

        compressed_list = compress_batch(batch_tokens, model, global_probs, device)

        for name, data in zip(file_names, compressed_list):
            compressed_index[name] = data

        total_raw_bytes += batch_tokens.size * batch_tokens.itemsize
        print(f"  {b_end}/{len(examples)}", end="\r", flush=True)

    print()

    # Stats before zip overhead
    total_compressed = sum(len(v) for v in compressed_index.values())
    raw_tokens_bits  = len(compressed_index) * 1200 * 128 * 10
    raw_bytes_tokens = raw_tokens_bits // 8
    print(f"Compressed data:  {raw_bytes_tokens/1e6:.1f} MB raw  →  "
          f"{total_compressed/1e6:.1f} MB  "
          f"({raw_bytes_tokens/total_compressed:.2f}× data-only)")

    # Build submission zip
    print(f"Writing {args.output} …")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as zf:
        # Python scripts (no compression: text already small)
        zf.write(ROOT / "decompress.py", "decompress.py")
        zf.write(ROOT / "training" / "__init__.py", "training/__init__.py")
        zf.write(ROOT / "training" / "model.py", "training/model.py")
        zf.write(ROOT / "training" / "coder.py", "training/coder.py")

        # Float16 model weights (same bytes that were used for encoding above)
        f16_buf.seek(0)
        zf.writestr("model_weights.pt", f16_buf.read())

        # Global frequency table
        zf.write(args.global_freq, "global_freq.npy")

        # Compressed token bitstreams
        pkl_buf = io.BytesIO()
        pickle.dump(compressed_index, pkl_buf)
        zf.writestr("compressed_data.pkl", pkl_buf.getvalue())

    zip_size = os.path.getsize(args.output)
    ratio    = raw_bytes_tokens / zip_size
    print(f"Zip size: {zip_size/1e6:.1f} MB")
    print(f"Overall compression ratio: {ratio:.2f}×")
    if ratio >= 3.5:
        print("✓ Meets the ≥3.5× target!")
    else:
        print(f"✗ Below 3.5× — consider training longer or increasing model capacity.")


if __name__ == "__main__":
    main()
