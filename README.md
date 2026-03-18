# video_compressor

Lossless compression of [commaVQ](https://huggingface.co/datasets/commaai/commavq) dashcam token sequences, targeting ≥ 3.5× compression ratio.

## What this repo currently uses

Active pipeline (kept and maintained):

- `training/train_global.py` — trains the next-frame model
- `model.py` — 4.48M parameter transformer (`8` frames context → next frame logits)
- `coder.py` — range-coding wrappers (`constriction`)
- `compress.py` — builds submission zip
- `decompress.py` — reconstructs tokens during evaluation
- `test/evaluate.sh` + `test/evaluate.py` — official-style verification

Legacy/experimental code has been moved to `legacy/`.

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional (avoid HF rate-limit warnings):

```bash
export HF_TOKEN=your_token_here
```

### 2) Train

Full dataset on a strong GPU:

```bash
python training/train_global.py \
  --shards 0 38 \
  --val-shards 38 40 \
  --epochs 10 \
  --batch 512 \
  --device auto \
  --workers 16
```

Checkpoints are saved during training:

- rolling: `resource/checkpoints/step_*.pt` (latest 3)
- per-epoch: `resource/checkpoints/epoch_*.pt`
- best: `resource/checkpoints/best.pt` (+ `resource/model.pt`)

Resume after interruption:

```bash
python training/train_global.py --auto-resume --shards 0 38 --epochs 10 --batch 512
```

### 3) Build submission zip

```bash
python compress.py --model resource/model.pt --output submission.zip --device auto
```

### 4) Evaluate

```bash
bash test/evaluate.sh submission.zip
```

## Outputs

Training artifacts:

- `resource/model.pt` (best float32 weights)
- `resource/model_f16.pt` (best float16 weights)
- `resource/global_freq.npy` (global token prior)
- `resource/checkpoints/*.pt` (resume checkpoints)

Submission:

- `submission.zip` (or the path passed to `--output`)

## Expected compression

With full-data training (shards `0..37`) and good convergence, expected ratio is typically around **3.3× to 4.0×**.  
Untrained or partially trained models usually stay near **1.0×–2.0×**.

## Notes

- Training and evaluation load data from Hugging Face (`commaai/commavq`) directly.
- Manual shard downloads are optional and not required for correctness.
- A deeper implementation breakdown is in `docs/TECHNICAL_OVERVIEW.md`.
