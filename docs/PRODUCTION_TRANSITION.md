# Python Evaluator + Rust Runtime Plan

This project keeps Python entrypoints for evaluator compatibility while allowing
entropy coding to run in Rust for production workloads.

## Current Structure

- `compress.py`: model inference + entropy coding orchestration.
- `decompress.py`: evaluator entrypoint + autoregressive decode loop.
- `coder.py`: frame encoder/decoder implementation used by both scripts.
- `model.py`, `training/train_global.py`: predictor architecture and training.
- `src/`: Rust rANS implementation and CLI binary.

## New Structure (Implemented)

- **Python interface remains unchanged**
  - `compress.py` and `decompress.py` still import `FrameEncoder/FrameDecoder`
    from `coder.py`.
  - Evaluator still runs `decompress.py` in Python.

- **Pluggable entropy backend in `coder.py`**
  - `CODEC_BACKEND=constriction` (default): existing Python `constriction` path.
  - `CODEC_BACKEND=rust`: calls Rust binary (`video_compressor`) for frame
    encode/decode through the same Python API.
  - `RUST_CODEC_BIN`: path override for the Rust binary.

- **Model/training code untouched**
  - No changes to `model.py`.
  - No changes to `training/train_global.py`.
  - No retraining required to adopt this runtime split.

## Usage

Build Rust binary:

```bash
cargo build --release
```

Run with Rust entropy backend:

```bash
CODEC_BACKEND=rust RUST_CODEC_BIN=target/release/video_compressor python compress.py ...
CODEC_BACKEND=rust RUST_CODEC_BIN=target/release/video_compressor python decompress.py
```

Compress with Rust backend:

```bash 
CODEC_BACKEND=rust RUST_CODEC_BIN=target/release/video_compressor \
python3 compress.py \
  --model resource/model.pt \
  --global-freq resource/global_freq.npy \
  --output compression_challenge_submission.zip \
  --device cuda \
  --encode-batch 512 \
  --coder-threads 24
```

Evaluate with matching backend:

```bash 
CODEC_BACKEND=rust RUST_CODEC_BIN=target/release/video_compressor \
DECOMPRESS_DEVICE=cuda \
bash test/evaluate.sh compression_challenge_submission.zip
```

Run with default backend (recommended for reference parity):

```bash
python compress.py ...
python decompress.py
```
