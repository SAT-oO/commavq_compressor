# Production Transition Notes

This repository currently ships a Python-first evaluation path because the
challenge requires a Python `decompress.py` entrypoint.

## Current production baseline

- `compress.py` + `coder.py` + `decompress.py` use `constriction` in Python.
- `decompress.py` is evaluator-compatible and deterministic-aware.
- `model.py` / `training/train_global.py` remain unchanged by runtime tuning.

## Rust status

- `src/` contains a Rust rANS implementation and CLI.
- The Rust path is kept for future production hardening, benchmarking, or service
  deployment, but it is not required by the official Python evaluator flow.

## Practical recommendation

For challenge submissions, keep the Python evaluator path as canonical:

```bash
python compress.py ...
bash test/evaluate.sh compression_challenge_submission.zip
```

Use Rust work as an optional follow-up optimization track after submission
stability is locked.
