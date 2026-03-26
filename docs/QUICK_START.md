## Quick start

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Optional (avoid Hugging Face unauthenticated warnings):

```bash
export HF_TOKEN=your_token_here
```

If running remote env:

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Train

Final training profile (requested):

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

Resume after interruption:

```bash
python training/train_global.py --auto-resume \
  --shards 0 38 \
  --epochs 40 \
  --batch 192 \
  --workers 16 \
  --prefetch-factor 4
```

_Note: use `tmux` if uninterrupted training is desired._


Checkpoint outputs:

- rolling: `resource/checkpoints/step_*.pt` (latest 3)
- per-epoch: `resource/checkpoints/epoch_*.pt`
- best: `resource/checkpoints/best.pt`
- exported best weights: `resource/model.pt`, `resource/model_f16.pt`

### 3. Build submission zip

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python compress.py \
  --model resource/model.pt \
  --global-freq resource/global_freq.npy \
  --splits data-0000.tar.gz data-0001.tar.gz \
  --output compression_challenge_submission.zip \
  --device cuda \
  --encode-batch 512 \
  --coder-threads 24
```

### 4. Evaluate

```bash
DECOMPRESS_DEVICE=cuda DECODE_BATCH=512 \
bash test/evaluate.sh compression_challenge_submission.zip
```

### 5. If evaluator path fails due to decode mismatch

Patch only the decompressor in an already-built zip (no recompression):

```bash
zip -u compression_challenge_submission.zip decompress.py
DECOMPRESS_DEVICE=cuda DECODE_BATCH=512 \
bash test/evaluate.sh compression_challenge_submission.zip
```
