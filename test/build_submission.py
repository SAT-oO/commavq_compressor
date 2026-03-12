#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "training" / "transition_topk_model.npz"
DEFAULT_ZIP = ROOT / "compression_challenge_submission.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the default model if needed and build a challenge-compatible submission zip."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained model. If missing and --train-if-missing is set, it will be created.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ZIP,
        help="Path to the submission zip to create.",
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train the default model when the model path does not exist.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k value to use if training is needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit passed through to training and compression.",
    )
    parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Disable the optional Rust bit-packing helper during compression.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    output_path = resolve_path(args.output)

    if not model_path.exists():
        if not args.train_if_missing:
            raise SystemExit(
                f"model not found: {model_path}. "
                "Run `python3 training/train_global.py` first or pass `--train-if-missing`."
            )

        train_cmd = [
            sys.executable,
            str(ROOT / "training" / "train_global.py"),
            "--output",
            str(model_path),
            "--top-k",
            str(args.top_k),
        ]
        if args.limit is not None:
            train_cmd.extend(["--limit", str(args.limit)])
        run(train_cmd)

    compress_cmd = [
        sys.executable,
        str(ROOT / "compress.py"),
        "--model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    if args.limit is not None:
        compress_cmd.extend(["--limit", str(args.limit)])
    if args.no_rust:
        compress_cmd.append("--no-rust")
    run(compress_cmd)

    print(f"submission ready: {output_path}")


if __name__ == "__main__":
    main()
