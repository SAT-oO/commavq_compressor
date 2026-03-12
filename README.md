# CommaVQ Token Compressor

Lossless temporal compression pipeline for commaVQ video tokens, built to turn already-compressed autonomous-driving latents into a smaller, challenge-ready submission archive.

## Why this project matters

- Compresses long-horizon video-token datasets without changing a single token.
- Combines sequence modeling, entropy coding, deterministic decoding, and evaluation-compatible packaging in one workflow.
- Demonstrates practical ML systems engineering: model export, portable runtime inference, challenge integration, and reproducible compression experiments.

## What it does

This project targets the `commaai/commavq` compression challenge, where the evaluator expects a single zip file containing:

- compressed data for the first two dataset splits
- a bundled `decompress.py`
- any additional source files needed to reconstruct the original token arrays exactly

The current implementation uses:

- a temporal mixture predictor trained over prior frames
- arithmetic/range-style entropy coding via `constriction`
- packed 10-bit storage for warmup context
- a pure-Python decompression path that remains compatible with the official evaluator

## Quick Start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the model:

```bash
python3 training/train_global.py
```

Build a submission archive:

```bash
python3 test/build_submission.py --train-if-missing
```

Evaluate with the untouched challenge script:

```bash
bash test/evaluate.sh compression_challenge_submission.zip
```

Get a fast small-sample estimate instead of full evaluation:

```bash
python3 estimate_sample.py --per-shard 32
```

## Repository Map

- `codec/`: binary format, shard reading, and 10-bit packing helpers
- `model/`: temporal predictive model definition and exported runtime weights
- `runtime/`: portable entropy-coding runtime helpers used by compression and decompression
- `training/`: model training entrypoint
- `test/`: official evaluation scripts plus a local submission builder

## Technical Details

The concise overview above is meant for fast reading. A deeper architecture and file-by-file walkthrough lives in `docs/TECHNICAL_OVERVIEW.md`.