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
    Decompress all samples sample-by-sample, frame-by-frame.
    """
    file_names  = list(compressed_index.keys())
    comp_bytes  = [compressed_index[n] for n in file_names]
    N           = len(file_names)

    global_tile = np.broadcast_to(
        global_probs[None, :], (TOKENS_PER_FRAME, len(global_probs))
    ).astype(np.float32)

    print(f"Decoding {N} samples on {device} …")

    for i, name in enumerate(file_names):
        dec = FrameDecoder(comp_bytes[i])
        num_frames = dec.n_frames
        tokens = np.zeros((num_frames, TOKENS_PER_FRAME), dtype=np.int16)

        for t in range(num_frames):
            if t == 0:
                probs = global_tile
            else:
                ctx = build_context_batch(tokens[None], t)      # (1, T, S)
                x = torch.tensor(ctx.astype(np.int64), device=device)
                with torch.no_grad():
                    logits = model(x)                           # (1, S, V)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            decoded = dec.decode_frame(probs)
            tokens[t] = decoded.astype(np.int16)

        out_path = output_dir / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Reshape back to original (num_frames, 8, 16) dataset format.
        np.save(out_path, tokens.reshape(num_frames, 8, 16))
        print(f"  {i+1}/{N}", end="\r", flush=True)

    print()

    print(f"Saved {N} arrays to {output_dir}")


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
        compressed_index = pickle.load(f)
    print(f"Compressed index: {len(compressed_index)} entries")

    decompress_all(compressed_index, model, global_probs, device, OUTPUT_DIR)


if __name__ == "__main__":
    main()
