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

Baseline full-data run (~2.93x compression rate)

```bash
python training/train_global.py \
  --shards 0 38 \
  --val-shards 38 40 \
  --epochs 16 \
  --batch 256 \
  --device auto \
  --workers 16 \
  --prefetch-factor 4
```

More ambitious run (longer training, ~3.05x compression rate):

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

_Note: use `tmux` if uninterrputed training is desired_


Checkpoint outputs:

- rolling: `resource/checkpoints/step_*.pt` (latest 3)
- per-epoch: `resource/checkpoints/epoch_*.pt`
- best: `resource/checkpoints/best.pt`
- exported best weights: `resource/model.pt`, `resource/model_f16.pt`

### 3. Build submission zip

```bash
python compress.py --model resource/model.pt --output submission.zip --device auto
```

### 4. Evaluate

```bash
bash test/evaluate.sh submission.zip
```
