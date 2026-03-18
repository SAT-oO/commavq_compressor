#!/bin/bash
# clean.sh — wipe all training artifacts so you can start fresh on the cloud.
# Safe to run multiple times. Does NOT touch source code or dataset tar.gz files.
set -e
cd "$(dirname "$0")"

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

# 5. HuggingFace Arrow cache (decoded dataset cache — regenerated automatically
#    from the .tar.gz shards on next run, so safe to delete)
rm -rf resource/dataset/commaai___commavq/
rm -rf resource/dataset/.cache/
rm -f  resource/dataset/*.lock
echo "  removed HuggingFace Arrow + lock cache"

echo ""
echo "=== Clean complete. Files kept ==="
echo "  resource/dataset/data-*.tar.gz   (raw shards — do NOT delete, re-download is slow)"
echo "  resource/tokens.npy              (sample reference, not used by training)"
echo "  resource/dataset_download.py"
echo "  All source code (*.py, *.rs)"
echo ""
echo "=== Ready for cloud training. Next steps ==="
echo "  1. Copy this repo to cloud:  rsync -av --exclude='.venv' . user@cloud:/workspace/commavq/"
echo "  2. On cloud:                 pip install -r requirements.txt"
echo "  3. Train:                    python training/train_global.py --shards 0 38 --val-shards 38 40 --epochs 10 --batch 128 --device auto"
echo "  4. Compress:                 python compress.py --model resource/model.pt --output submission.zip"
echo "  5. Evaluate:                 bash test/evaluate.sh submission.zip"
