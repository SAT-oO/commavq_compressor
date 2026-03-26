# Commavq Compression Technical Overview

## Scope

This repository implements a lossless neural+entropy codec for the official
evaluation subset of [commaai/commavq](https://huggingface.co/datasets/commaai/commavq):
`data-0000.tar.gz` and `data-0001.tar.gz` (5000 samples).

Observed with current final settings:

- Data-only rate: ~`3.05x`
- Overall submission rate: ~`2.96x` (displayed as `3.0x` by evaluator)

## Model (`model.py`)

`NextFramePredictor` is a transformer encoder that predicts the next-frame token
distribution from 8 prior frames:

- Input shape: `(B, 8, 128)`
- Output shape: `(B, 128, 1024)`
- Parameter count: `4,483,584` (~4.48M)
- Embeddings: token + temporal + decomposed 2D spatial (`row + col`)

Compression depends on probability quality because entropy coding cost is
`-log2 p(symbol)`.

## Entropy coder (`coder.py`)

`coder.py` uses `constriction.stream.queue.RangeEncoder/RangeDecoder` with
categorical models:

- Encoder API: `FrameEncoder.encode_frame(tokens, probs)`
- Decoder API: `FrameDecoder.decode_frame(probs)`
- Stream format: single continuous range stream per sample with frame-count
  header.

Probability handling includes explicit rounding + clipping + renormalization to
reduce numeric drift.

## Training (`training/train_global.py`)

Main responsibilities:

- Load shards via Hugging Face `datasets`
- Build `(context, target)` pairs
- Train with AdamW + cosine schedule
- Export model and checkpoints

Primary training profile used for final docs:

```bash
python training/train_global.py \
  --shards 0 38 \
  --val-shards 38 40 \
  --epochs 40 \
  --batch 192 \
  --device auto \
  --workers 16 \
  --prefetch-factor 4
```

Checkpoint outputs:

- rolling: `resource/checkpoints/step_*.pt` (latest 3)
- epoch: `resource/checkpoints/epoch_*.pt`
- best: `resource/checkpoints/best.pt`
- exported: `resource/model.pt`, `resource/model_f16.pt`

## Compression (`compress.py`)

Compression workflow:

1. Load model and `global_freq.npy`
2. Round-trip model through float16 for decode parity
3. Predict probabilities and entropy-code tokens
4. Package submission zip

Submission archive contains:

- `decompress.py`
- `model.py`
- `coder.py`
- `model_weights.pt`
- `global_freq.npy`
- `compressed_data.pkl`

## Decompression (`decompress.py`)

Decompression workflow:

1. Load bundled model weights + global prior
2. Decode range stream using predicted probabilities
3. Write outputs into `OUTPUT_DIR` using exact file keys expected by evaluator

Important runtime controls:

- `DECOMPRESS_DEVICE`: force `cuda` or `cpu`
- `DECODE_BATCH`: decode batch size (used to match encode inference shape when needed)

## Evaluation (`test/evaluate.sh`, `test/evaluate.py`)

Evaluation process:

1. Unzip submission
2. Run bundled `decompress.py`
3. Compare all decoded tokens against Hugging Face ground truth
4. Report compression ratio

Ratio formula used by evaluator:

`(5000 * 1200 * 128 * 10 / 8) / submission_zip_size`

## Active file map

- `training/train_global.py`
- `model.py`
- `coder.py`
- `compress.py`
- `decompress.py`
- `test/evaluate.sh`
- `test/evaluate.py`

Legacy/experimental code remains under `legacy/`.
