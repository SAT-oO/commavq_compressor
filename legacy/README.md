# Legacy Code

This directory contains code paths that are **not used** by the current training/compression/decompression pipeline.

Moved here during repository cleanup to keep the active root workflow minimal and production-focused.

Contents:

- `legacy/codec/` — older codec format helpers
- `legacy/runtime/` — older entropy runtime helpers
- `legacy/model/temporal.py` — previous temporal model
- `legacy/estimate_sample.py` — old sampling estimator
- `legacy/test/build_submission.py` — old submission convenience wrapper
- `legacy/resource/dataset_download.py` and `legacy/resource/tokens.npy` — manual dataset/sample utilities

Active pipeline files are documented in:

- `README.md`
- `docs/TECHNICAL_OVERVIEW.md`

