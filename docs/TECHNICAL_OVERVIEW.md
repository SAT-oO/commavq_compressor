# Technical Overview

## Goal

This repository implements a lossless compressor for commaVQ token sequences that remains compatible with the official `commaai/commavq/compression` evaluation flow.

The project is organized around four responsibilities:

- learn a hierarchical temporal predictor over token frames
- convert copy modes and novel-token probability distributions into a compressed bitstream
- reconstruct the original tokens exactly during decompression
- package the result into the archive shape expected by the challenge evaluator

## Architecture

### `codec/`

`codec/` contains the reusable codec mechanics.

- `codec/bits.py`
  Packs and unpacks token arrays into dense 10-bit form. It optionally uses the Rust helper binary for faster pack/unpack.

- `codec/dataset.py`
  Reads local `.tar.gz` evaluation shards directly and yields `(file_name, tokens)` records. It also provides fast record counting and per-shard sampling helpers for small experiments.

- `codec/format.py`
  Defines the archive format stored inside `data.bin`. Warmup frames are stored in packed 10-bit form. For later frames, the codec first entropy-codes a small copy-mode stream and then entropy-codes only the novel tokens that are not explained by lagged copies.

### `model/`

- `model/temporal.py`
  Defines `TemporalMixtureModel`, a deterministic hierarchical predictor. It combines:
  - copy-mode priors for several temporal lags
  - global token priors
  - per-row and per-position token priors
  - cluster-conditioned token priors
  - lagged transition distributions
  - row-conditioned lagged transition distributions

  At inference time it predicts:
  - a copy mode for each of the `128` positions
  - a novel-token distribution for positions that are not copied from prior lags

### `runtime/`

- `runtime/entropy.py`
  Provides the portable entropy-coding runtime. It uses `constriction` for categorical range coding and includes portable frame-by-frame encoder/decoder helpers used by both compression and decompression.

### Top-level entrypoints

- `training/train_global.py`
  Trains and exports the temporal model as `training/temporal_mixture_model.npz`.

- `compress.py`
  Builds a challenge-compatible submission zip by encoding the target shards and bundling the decompression runtime.

- `decompress.py`
  The script that the official evaluator executes after unzipping the submission. It reconstructs the extensionless NumPy token files into `OUTPUT_DIR`.

- `estimate_sample.py`
  Runs a balanced small-sample experiment on the first two evaluation shards and reports an estimated full-dataset compression rate.

- `test/build_submission.py`
  Local convenience wrapper that trains the model if needed and then invokes `compress.py`.

## Compression Flow

1. Read token arrays from the evaluation shards.
2. Store the first `warmup_frames` raw in packed 10-bit form.
3. For each later frame:
   - predict a copy mode for each position
   - entropy-code the copy modes
   - for positions marked as novel, generate token distributions from prior context
   - entropy-code only those novel tokens
4. Store all segments inside a single `data.bin` archive plus a compact exported model.
5. Bundle `decompress.py` and its required runtime files into the submission zip.

## Decompression Flow

1. Load `model.npz`.
2. Read `data.bin`.
3. Restore warmup frames from packed 10-bit storage.
4. Reconstruct later frames sequentially by:
   - recomputing copy-mode probabilities from already-decoded context
   - decoding copy modes
   - copying tokens from prior lags when the mode indicates a copy
   - decoding only the remaining novel tokens with the same categorical model family
5. Write exact token arrays to the output location expected by the official evaluator.

## Current Design Trade-offs

- The model is deterministic and lightweight, which keeps the decompression runtime portable.
- The entropy coder is real probability-driven coding, not a generic backend compressor like `lzma`.
- The explicit copy-mode split is designed to capture cheap temporal redundancy before novel-token coding.
- The predictor is still simpler than a transformer-based model, so this repo is structured to make future model upgrades straightforward.

## Recommended Workflow

Full challenge build:

```bash
python3 test/build_submission.py --train-if-missing
bash test/evaluate.sh compression_challenge_submission.zip
```

Quick experimental estimate:

```bash
python3 estimate_sample.py --per-shard 32
```
