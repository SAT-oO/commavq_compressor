#!/usr/bin/env python3
"""
Quick encode/decode parity smoke test.

Runs the current `compress` + `decompress` pipeline on a small token sample and
asserts bit-exact reconstruction. Use this before expensive full compress runs.
"""

from pathlib import Path
import tempfile
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compress import compress_batch
from decompress import decompress_all
from model import NextFramePredictor


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    sample_path = Path("legacy/resource/tokens.npy")
    if sample_path.exists():
        tokens = np.load(sample_path).astype(np.int16)
    else:
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 1024, size=(1200, 128), dtype=np.int16)

    # Keep this test lightweight while still long enough to catch drift.
    n_frames = 120
    tokens = tokens[:n_frames]
    batch_tokens = tokens[None]  # (1, T, 128)

    model = NextFramePredictor().eval()
    global_counts = np.bincount(tokens.reshape(-1), minlength=1024).astype(np.float32)
    global_probs = global_counts + 1.0
    global_probs /= global_probs.sum()

    compressed = compress_batch(batch_tokens, model, global_probs, device="cpu")[0]
    compressed_index = {"sample/token.npy": compressed}

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        decompress_all(compressed_index, model, global_probs, "cpu", out_dir)
        recon = np.load(out_dir / "sample/token.npy").reshape(n_frames, 128).astype(np.int16)

    ok = np.array_equal(tokens, recon)
    print(f"roundtrip_ok={ok} frames={n_frames} compressed_bytes={len(compressed)}")
    if not ok:
        raise SystemExit("Round-trip mismatch: encode/decode are not verbatim.")


if __name__ == "__main__":
    main()
