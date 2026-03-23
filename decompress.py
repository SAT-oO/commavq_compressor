#!/usr/bin/env python3
"""
Decompress commavq token sequences from a submission zip.

evaluate.sh calls this script as:
    OUTPUT_DIR=<decompressed_dir> python decompress.py

The script reads all auxiliary files from the same directory as itself
(SCRIPT_DIR), writes decompressed .npy arrays to OUTPUT_DIR.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR", SCRIPT_DIR))

sys.path.insert(0, str(SCRIPT_DIR))

from coder import FrameDecoder
from model import (
    CONTEXT_FRAMES,
    TOKENS_PER_FRAME,
    NextFramePredictor,
    build_context_batch,
)

DECODE_BATCH = 256   # samples decoded in parallel per model forward pass


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_weights(weights_path: Path, device: str) -> NextFramePredictor:
    model = NextFramePredictor().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    # Cast float16 weights back to float32 for accurate inference.
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def decompress_all(
    compressed_index: dict,
    model: NextFramePredictor,
    global_probs: np.ndarray,
    device: str,
    output_dir: Path,
) -> None:
    """
    Decompress all samples, batching model inference across samples at each
    time step to amortise GPU kernel launch overhead.

    Because decoding is sequential within each sample (frame t's decoded
    tokens are the context for frame t+1), we must maintain per-sample
    decoder state across time steps.
    """
    file_names  = list(compressed_index.keys())
    comp_bytes  = [compressed_index[n] for n in file_names]
    N           = len(file_names)

    global_tile = np.broadcast_to(
        global_probs[None, :], (TOKENS_PER_FRAME, len(global_probs))
    ).astype(np.float32)

    # Create one FrameDecoder per sample; read frame count from each header.
    decoders = [FrameDecoder(b) for b in comp_bytes]
    num_frames = decoders[0].n_frames          # all samples must have same length

    # Allocate decoded token buffer (all samples in memory at once).
    all_tokens = np.zeros((N, num_frames, TOKENS_PER_FRAME), dtype=np.int16)

    print(f"Decoding {N} samples × {num_frames} frames on {device} …")

    for t in range(num_frames):
        # ── Model inference (batched over all N samples) ───────────────────
        if t == 0:
            # First frame: use global marginal prior.
            probs_all = np.stack([global_tile] * N)    # (N, S, V)
        else:
            probs_parts = []
            for b_start in range(0, N, DECODE_BATCH):
                b_end = min(b_start + DECODE_BATCH, N)
                ctx   = build_context_batch(all_tokens, t)  # (N, T, S)
                ctx_b = ctx[b_start:b_end]
                x     = torch.tensor(ctx_b.astype(np.int64), device=device)
                with torch.no_grad():
                    logits = model(x)                        # (B, S, V)
                    probs_parts.append(
                        torch.softmax(logits, dim=-1).cpu().numpy()
                    )
            probs_all = np.concatenate(probs_parts, axis=0)  # (N, S, V)

        # ── Per-sample decode (sequential but cheap) ───────────────────────
        for i in range(N):
            decoded = decoders[i].decode_frame(probs_all[i])     # (S,) int32
            all_tokens[i, t] = decoded.astype(np.int16)

        if (t + 1) % 100 == 0:
            print(f"  frame {t+1}/{num_frames}", end="\r", flush=True)

    print()

    # Save decompressed arrays.
    for i, name in enumerate(file_names):
        out_path = output_dir / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Reshape back to original (num_frames, 8, 16) dataset format.
        np.save(out_path, all_tokens[i].reshape(num_frames, 8, 16))

    print(f"Saved {N} arrays to {output_dir}")


def main() -> None:
    device = select_device()
    print(f"Device: {device}")

    model = load_model_weights(SCRIPT_DIR / "model_weights.pt", device)
    print(f"Model loaded ({model.param_count():,} parameters)")
    try:
        model = torch.compile(model)
        print("  torch.compile() applied")
    except Exception:
        pass

    global_probs = np.load(SCRIPT_DIR / "global_freq.npy").astype(np.float32)
    global_probs = np.clip(global_probs, 1e-6, None)
    global_probs /= global_probs.sum()

    with open(SCRIPT_DIR / "compressed_data.pkl", "rb") as f:
        compressed_index = pickle.load(f)
    print(f"Compressed index: {len(compressed_index)} entries")

    decompress_all(compressed_index, model, global_probs, device, OUTPUT_DIR)


if __name__ == "__main__":
    main()
