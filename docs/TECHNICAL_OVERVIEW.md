# Commavq Compression Challenge 

## Overview 

Goal is to compress CommaVQ token sequences fetched from [HuggingFace](https://huggingface.co/datasets/commaai/commavq) loselessly as part of CommaAI Compression Challenge. 

Overall performance comes out to be **~3.05x compression rate**. 

---

## Architecture

### 1. Predictor: `model.py`

`NextFramePredictor` is a transformer encoder that predicts next-frame token probabilities.

- input: `B x 8 x 128` token IDs (`8` prior frames, `128` positions/frame)
- output: `B x 128 x 1024` logits (`1024` token classes per position)
- parameter count: ~4.48M
- positional scheme:
  - temporal embedding for frame index in the context window
  - decomposed spatial embedding (`row + col`) for the 8x16 grid

Why this model helps compression:

- range coding cost for a symbol is `-log2(p(symbol))` bits
- better probability calibration from temporal context lowers average bits/token
- lower bits/token yields higher compression ratio, while still being lossless

### 2. Entropy coder: `coder.py`

`coder.py` wraps `constriction` range coding with a simple API:

- `FrameEncoder.encode_frame(tokens, probs)`
- `FrameDecoder.decode_frame(probs)`

Both encoder and decoder use the same model family (`Categorical`) and the same quantized model weights, ensuring deterministic decode.

### 3. Training: `training/train_global.py`

Responsibilities:

- load selected commaVQ shards via Hugging Face `datasets`
- sample random `(context, target)` pairs from each video clip
- train with AdamW + cosine schedule
- write:
  - `resource/model.pt`
  - `resource/model_f16.pt`
  - `resource/global_freq.npy`
  - multi-tier checkpoints under `resource/checkpoints/`

Performance features:

- CUDA auto-selection, BF16 mixed precision (optional)
- `torch.compile` (optional)
- configurable worker/prefetch settings for dataloaders
- CPU threading setup for high-core machines

Checkpoint tiers:

- rolling `step_*.pt` (latest 3)
- epoch `epoch_*.pt` (kept)
- best `best.pt` + exported model weights

Resume options:

- `--resume-from <path>`
- `--auto-resume`

### 4. Compression packaging: `compress.py`

Workflow:

1. load trained model and global prior
2. quantize model to float16 bytes and reload for deterministic parity with decompressor
3. stream dataset samples, predict per-frame probabilities in chunks
4. entropy-code each frame
5. write submission zip containing:
   - `decompress.py`
   - `model.py`
   - `coder.py`
   - `model_weights.pt`
   - `global_freq.npy`
   - `compressed_data.pkl`

Memory behavior:

- current implementation avoids materializing full-probability tensors for all frames at once
- per-sample chunked prediction reduces peak RAM usage during compression

### 5. Decompression: `decompress.py`

Workflow at evaluation time:

1. load bundled float16 model weights
2. load global prior and compressed bitstreams
3. decode frames sequentially using the same predicted distributions used for encoding
4. write `.npy` token files into `OUTPUT_DIR`

Lossless guarantee:

- arithmetic/range decoding is exact if model outputs and symbol streams match
- this is why compression path uses the same quantized weight representation shipped in the zip

### 6. Evaluation: `test/evaluate.sh`, `test/evaluate.py`

- unzips submission
- runs bundled `decompress.py`
- validates reconstructed tokens against ground truth
- prints compression ratio

---

## Data Flow Summary

1. **Train:** raw commaVQ shards -> trained predictor (`model.pt`)
2. **Compress:** predictor + tokens -> entropy-coded streams + model bundle (`submission.zip`)
3. **Decompress:** bundled model + streams -> exact token reconstruction
4. **Evaluate:** compare reconstruction + compute ratio

---

## Repository Map (active files)

- `training/train_global.py`
- `model.py`
- `coder.py`
- `compress.py`
- `decompress.py`
- `test/evaluate.py`
- `test/evaluate.sh`
- `src/` (optional Rust rANS implementation; not required by Python pipeline)

Legacy modules are in `legacy/`.

---

## Operational Notes

- Training/compression pull data from Hugging Face directly; manual shard download is optional.
- For large runs, keep checkpoints and use `--auto-resume`.
- If `compress.py` is killed on CPU, reduce memory pressure with `--frame-chunk` and/or run on CUDA.

---

## Cleanup Tip

At the end of experiments, two maintenance scripts in `test/` can help:

- `test/clean.sh`  
  Local repository cleanup: removes generated training artifacts, checkpoints, zip outputs, and caches.

- `test/clean.py`  
  Hugging Face model-repo cleanup helper: removes accidentally uploaded root-level checkpoint files.
