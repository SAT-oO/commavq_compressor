#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <path_to_submission.zip> [path_to_smoke_gt.npy]"
  exit 1
fi

ZIP_FILE="$(realpath "$1")"
GT_FILE="${2:-./test/smoke_gt.npy}"
GT_FILE="$(realpath "$GT_FILE")"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(realpath "$DIR/..")"
DECOMPRESSED_DIR="$DIR/compression_challenge_submission_decompressed"

rm -rf "$DECOMPRESSED_DIR"
unzip -o "$ZIP_FILE" -d "$DECOMPRESSED_DIR" >/dev/null

OUTPUT_DIR="$DECOMPRESSED_DIR" \
DECOMPRESS_DEVICE="${DECOMPRESS_DEVICE:-cpu}" \
python "$DECOMPRESSED_DIR/decompress.py"

PACKED_ARCHIVE="$ZIP_FILE" \
UNPACKED_ARCHIVE="$DECOMPRESSED_DIR" \
SMOKE_GT="$GT_FILE" \
python "$DIR/evaluate_smoke.py"
