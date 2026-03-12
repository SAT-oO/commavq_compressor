# video_compressor

First draft of a lossless compressor for the commaVQ token dataset.

## Challenge Contract

The official challenge in `commaai/commavq/compression` evaluates a submission by running:

```bash
./compression/evaluate.sh path_to_submission.zip
```

The submission zip must contain:

- compressed data for the first two dataset splits
- a Python script named `decompress.py`
- any extra project files needed by `decompress.py` that are not assumed to exist already

This repo is now aligned to that contract:

- `compress.py` emits a single submission zip.
- The zip contains `decompress.py`, `data.bin`, `model.npz`, and the shared codec modules.
- `decompress.py` reconstructs extensionless NumPy files in `OUTPUT_DIR`, matching the evaluator expectation exactly.
- `test/evaluate.sh` and `test/evaluate.py` are kept aligned with the official judging scripts rather than being repurposed as local build helpers.

## Current Draft

This repository now uses a hybrid layout:

- `training/train_global.py`: trains a compact global transition model in Python.
- `compress.py`: builds a submission zip with `data.bin`, `model.npz`, and a standalone `decompress.py`.
- `decompress.py`: reconstructs extensionless NumPy token files compatible with `test/evaluate.py`.
- `src/main.rs`: optional Rust helper for fast 10-bit pack/unpack used by the Python codec when a local binary is available.

The current predictor is intentionally conservative for a first draft:

- No intra-frame autoregression.
- Predicts each token from the same token position in the previous frame plus global priors.
- Encodes a top-k rank when the true token appears in the model shortlist.
- Falls back to raw 10-bit escapes for misses.

This is not expected to be leaderboard-optimal yet, but it creates a real end-to-end path that can be iterated toward stronger temporal models.

## Files

- `codec/dataset.py`: reads local `resource/dataset/*.tar.gz` shards directly.
- `codec/model.py`: compact global transition-top-k model.
- `codec/bits.py`: Python pack/unpack with optional Rust acceleration.
- `codec/format.py`: shared binary format for compression and decompression.

## Quick Start

Set up a Python environment with the project dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a first model on the local evaluation shards:

```bash
python3 training/train_global.py
```

Build the optional Rust helper:

```bash
cargo build --release
```

Create a submission archive:

```bash
python3 test/build_submission.py --model training/transition_topk_model.npz
```

Run the repo evaluator:

```bash
test/evaluate.sh compression_challenge_submission.zip
```

## Notes

- The draft currently optimizes for a small, deterministic model and a working submission format.
- `decompress.py` is pure Python on purpose so the submission does not depend on a Rust toolchain.
- `test/build_submission.py` is the convenience entrypoint for local development; the evaluator remains the official challenge entrypoint.
- The main next upgrade path is replacing the transition shortlist model with a stronger temporal predictor while keeping the same archive format.