<<<<<<< Current (Your changes)
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

- a hierarchical temporal predictor with explicit copy-mode detection
- cluster-conditioned novel-token probability models over prior frames
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
=======
# video_compressor

Lossless compression of [commaVQ](https://huggingface.co/datasets/commaai/commavq) dashcam token sequences, targeting ≥ 3.5× compression ratio.

## Method

**Neural next-frame predictor + ANS/range entropy coding.**

Each video segment is a sequence of 1200 frames, each represented by 128 VQ tokens (8×16 spatial grid, values 0–1023, 10 bits raw).  The compressor:

1. **Predicts** the probability distribution over 1024 tokens for every spatial position in frame *t*, given the previous 8 frames as context.
2. **Entropy-codes** the actual token with that predicted distribution using a range coder.

Because consecutive dashcam frames are highly correlated, the predictor assigns high probability to the correct token, leading to far fewer bits per token than the raw 10-bit encoding.

### Architecture (NextFramePredictor)

| Component | Config |
|-----------|--------|
| Input | (B, T=8, 128) int token IDs |
| Token embedding | 1024 → 256 |
| Spatial pos (2-D decomposed) | row(8) + col(16) → 256 |
| Temporal pos | 8 → 256 |
| TransformerEncoder | 6 layers, d=256, 4 heads, FFN=768, Pre-LN |
| Output head | 256 → 1024 logits (last 128 positions) |
| **Total params** | **~4.5 M** |
| Float16 size | ~9 MB |

### Rust rANS coder (`src/rans_coder.rs`)

A standalone rANS (range Asymmetric Numeral Systems) implementation in Rust with:

- State space L = 2²³, renorm base b = 256 (byte-IO)
- Frequency precision M = 2¹⁶ = 65536
- LIFO encode / FIFO decode (encoded in reverse, decoded forward)

The Rust binary is an optional faster alternative to the Python/constriction path.

## Compression ratio analysis

| Predictor | bits/token | compression |
|-----------|-----------|-------------|
| Raw | 10.0 | 1.0× |
| Marginal frequency | ~9.7 | ~1.0× |
| 1st-order Markov (same position) | 4.31 | 2.3× |
| **Trained 4.5M transformer (T=8)** | **~2–3** | **~3.5–5×** |

The neural model exploits deep temporal and spatial context that simple Markov models cannot, achieving the target ≥ 3.5× compression after training.

## Usage

### 1. Install dependencies

```bash
pip install torch numpy constriction datasets huggingface_hub
```

### 2. Download dataset

```bash
python resource/dataset_download.py
```

### 3. Train the model

```bash
python training/train_global.py --shards 0 38 --epochs 5 --device auto
```

Optional: to avoid Hugging Face rate-limit warnings, set a token and re-run (get one at https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN=your_token_here
```

**Quick training run (fewer shards, less download):** To avoid downloading all 38 shards, train on 2 shards first (e.g. ~5k samples). Download is much smaller and faster:

```bash
python training/train_global.py --shards 0 2 --epochs 3 --device auto
```

**Note on `commaai/commavq-gpt2m`:** That Hugging Face model is a pre-trained **GPT-2 style causal LM** for *generating* driving token sequences. It is not used here. This repo uses the **commaVQ dataset** only and trains a smaller **frame-level** predictor (8 frames → next frame’s 128 tokens in one forward pass), which is faster for compression (1200 passes per video vs token-by-token).

Training outputs:
- `resource/model.pt` — best checkpoint (float32)
- `resource/model_f16.pt` — same weights in float16
- `resource/global_freq.npy` — marginal token frequency table

### 4. Compress to submission zip

```bash
python compress.py --model resource/model.pt --output submission.zip
```

### 5. Evaluate

```bash
bash test/evaluate.sh submission.zip
```

This runs `decompress.py` (from inside the zip) and then `test/evaluate.py` which verifies lossless round-trip and prints the compression rate.

### Optional: build Rust binary

```bash
cargo build --release
# Binary: target/release/video_compressor
```

## File layout

```
zeq/
├── model.py              # NextFramePredictor architecture
├── coder.py              # Range-coding wrappers (constriction)
├── compress.py           # Compression pipeline + zip builder
├── decompress.py         # Decompression pipeline (also in submission zip)
├── training/
│   └── train_global.py   # Training script
├── src/
│   ├── main.rs           # Rust CLI (encode / decode mode)
│   └── rans_coder.rs     # rANS implementation
├── resource/
│   ├── dataset/          # commaVQ shards (data-0000…data-0039.tar.gz)
│   ├── model.pt          # Trained model (after running train_global.py)
│   └── global_freq.npy   # Marginal frequency table
└── test/
    ├── evaluate.py
    └── evaluate.sh
```
>>>>>>> Incoming (Background Agent changes)
