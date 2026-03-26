# Commavq Compression Challenge

Lossless compression pipeline for [commaai/commavq](https://huggingface.co/datasets/commaai/commavq).

Current validated result on the official evaluation path (`data-0000` + `data-0001`):

- Data-only rate (`compressed_data.pkl`): ~`3.05x`
- Overall submission rate (`submission.zip`): ~`2.96x` (prints as `3.0x`)

## Active pipeline

- `training/train_global.py` — train predictor, export model/frequency files, write checkpoints.
- `model.py` — `NextFramePredictor` transformer (~4.48M params, 8-frame context).
- `coder.py` — deterministic `constriction` range coding wrappers.
- `compress.py` — build submission archive.
- `decompress.py` — evaluator-compatible reconstruction script.
- `test/evaluate.sh` + `test/evaluate.py` — official local validation path.

Legacy/unused code paths are kept in `legacy/`.

## Core artifacts

Training artifacts:

- `resource/model.pt`
- `resource/model_f16.pt`
- `resource/global_freq.npy`
- `resource/checkpoints/*.pt`

Submission archive contents:

- `decompress.py`
- `model.py`
- `coder.py`
- `model_weights.pt`
- `global_freq.npy`
- `compressed_data.pkl`

## Where to start

- Quick runbook: `docs/QUICK_START.md`
- Technical details: `docs/TECHNICAL_OVERVIEW.md`
