#!/usr/bin/env python3
"""
Decompress commavq token sequences from a submission zip.

evaluate.sh calls this script as:
    OUTPUT_DIR=<decompressed_dir> python decompress.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", SCRIPT_DIR))

sys.path.insert(0, str(SCRIPT_DIR))

from coder import FrameDecoder
from model import TOKENS_PER_FRAME, NextFramePredictor, build_context_batch


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def select_device() -> str:
    forced = os.environ.get("DECOMPRESS_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_weights(weights_path: Path, device: str) -> NextFramePredictor:
    model = NextFramePredictor().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
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
    decode_batch: int = 1,
) -> None:
    """Decompress in batches to match encoder inference shape when needed."""
    file_names = list(compressed_index.keys())
    comp_bytes = [compressed_index[n] for n in file_names]
    n_samples = len(file_names)

    global_tile = np.broadcast_to(
        global_probs[None, :], (TOKENS_PER_FRAME, len(global_probs))
    ).astype(np.float32)

    decode_batch = max(1, int(decode_batch))
    print(f"Decoding {n_samples} samples on {device} (decode_batch={decode_batch}) ...")

    done = 0
    for b_start in range(0, n_samples, decode_batch):
        b_end = min(b_start + decode_batch, n_samples)
        names_b = file_names[b_start:b_end]
        bytes_b = comp_bytes[b_start:b_end]
        batch_size = len(names_b)

        decoders = [FrameDecoder(b) for b in bytes_b]
        num_frames = decoders[0].n_frames
        tokens_b = np.zeros((batch_size, num_frames, TOKENS_PER_FRAME), dtype=np.int16)

        for t in range(num_frames):
            if t == 0:
                probs_all = np.broadcast_to(
                    global_tile[None, :, :],
                    (batch_size, TOKENS_PER_FRAME, len(global_probs)),
                )
            else:
                ctx = build_context_batch(tokens_b, t)
                x = torch.tensor(ctx.astype(np.int64), device=device)
                with torch.no_grad():
                    logits = model(x)
                    probs_all = torch.softmax(logits, dim=-1).cpu().numpy()

            for i in range(batch_size):
                decoded = decoders[i].decode_frame(probs_all[i])
                tokens_b[i, t] = decoded.astype(np.int16)

        for i, name in enumerate(names_b):
            out_path = output_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Save to exact evaluator key path (no ".npy" auto-suffix).
            with open(out_path, "wb") as f:
                np.save(f, tokens_b[i].reshape(num_frames, 8, 16))
            done += 1
            print(f"  {done}/{n_samples}", end="\r", flush=True)

    print()
    print(f"Saved {n_samples} arrays to {output_dir}")


def main() -> None:
    configure_determinism()

    device = select_device()
    print(f"Device: {device}")

    model = load_model_weights(SCRIPT_DIR / "model_weights.pt", device)
    print(f"Model loaded ({model.param_count():,} parameters)")

    global_probs = np.load(SCRIPT_DIR / "global_freq.npy").astype(np.float32)
    global_probs = np.clip(global_probs, 1e-6, None)
    global_probs /= global_probs.sum()

    with open(SCRIPT_DIR / "compressed_data.pkl", "rb") as f:
        packed = pickle.load(f)

    # Backward-compatible with both old and wrapped layouts.
    if isinstance(packed, dict) and "__data__" in packed:
        compressed_index = packed["__data__"]
        encode_batch_hint = int(packed.get("__meta__", {}).get("encode_batch", 1))
    else:
        compressed_index = packed
        encode_batch_hint = 1

    print(f"Compressed index: {len(compressed_index)} entries")
    decode_batch = int(os.environ.get("DECODE_BATCH", encode_batch_hint))

    decompress_all(
        compressed_index,
        model,
        global_probs,
        device,
        OUTPUT_DIR,
        decode_batch=decode_batch,
    )


if __name__ == "__main__":
    main()
