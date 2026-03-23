#!/bin/bash
# clean.sh — wipe generated artifacts while keeping source code.
# Safe to run multiple times.
set -e
cd "$(dirname "$0")/.."

echo "=== Cleaning training artifacts ==="

# 1. Trained model weights (bad/partial runs)
rm -f resource/model.pt
rm -f resource/model_f16.pt
echo "  removed model.pt + model_f16.pt"

# 2. Global frequency table (recomputed during training)
rm -f resource/global_freq.npy
echo "  removed global_freq.npy"

# 3. All checkpoints
rm -rf resource/checkpoints/
echo "  removed resource/checkpoints/"

# 4. Old submission zip(s)
rm -f compression_challenge_submission.zip submission.zip
echo "  removed submission zip(s)"

# 5. Dataset/cache artifacts (HF cache and any manually downloaded shards)
rm -rf resource/dataset/
echo "  removed resource/dataset/ cache + shard copies"

# 6. Python and Rust build caches
rm -rf __pycache__/ test/__pycache__/ training/__pycache__/ target/
echo "  removed build/cache folders"

echo ""
echo "=== Clean complete. Files kept ==="
echo "  All source code (*.py, *.rs, docs)"
echo "  Legacy code under legacy/"
echo ""
echo "=== Ready for cloud training. Next steps ==="
echo "  1. Copy this repo to cloud:  rsync -av --exclude='.venv' . user@cloud:/workspace/commavq/"
echo "  2. On cloud:                 pip install -r requirements.txt"
echo "  3. Train:                    python training/train_global.py --shards 0 38 --val-shards 38 40 --epochs 10 --batch 512 --device auto --workers 16"
echo "  4. Compress:                 python compress.py --model resource/model.pt --output submission.zip"
echo "  5. Evaluate:                 bash test/evaluate.sh submission.zip"
