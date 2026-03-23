# Commavq Video Compression Challenge

Lossless compression of [commaVQ](https://huggingface.co/datasets/commaai/commavq) HuggingFace dashcam token sequences for comma.ai challenge evaluation.

Overall performance of **3.05x overall compression** with training parameters specified in `docs`. 

Available on HuggingFace: [SAT-oO/commavq_compression](https://huggingface.co/SAT-oO/commavq_compression/tree/main)

## Active pipeline

The maintained end-to-end path is:

- `training/train_global.py` — train next-frame predictor + checkpoints
- `model.py` — 4.48M parameter transformer (`8` frame context)
- `coder.py` — range coding wrappers (`constriction`)
- `compress.py` — build submission zip
- `decompress.py` — reconstruct tokens in evaluator
- `test/evaluate.sh` + `test/evaluate.py` — correctness + ratio check

Legacy/unused approaches are in `legacy/`.

## Output artifacts

Training outputs:

- `resource/model.pt`
- `resource/model_f16.pt`
- `resource/global_freq.npy`
- `resource/checkpoints/*.pt`

Submission output:

- `submission.zip` (or custom `--output`)

## Notes

- Training and evaluation download data from Hugging Face directly (`commaai/commavq`).
- Manual local shard copies are optional.
- `compress.py` can be memory-heavy on CPU; prefer CUDA for packaging.
- Implementation details: `docs/TECHNICAL_OVERVIEW.md`.
- To replicate trainig workflow: see `docs/QUICK_START.md`
