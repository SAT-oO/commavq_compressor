#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from codec.dataset import count_shard_records, default_eval_shards, iter_shard_records
from codec.format import encode_records
from model.temporal import TemporalMixtureModel


SUBMISSION_FILES = [
    "decompress.py",
    "codec/bits.py",
    "codec/format.py",
    "model/temporal.py",
    "runtime/entropy.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a challenge-compatible commaVQ submission archive.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("training/temporal_mixture_model.npz"),
        help="Path to a trained predictor model exported by training/train_global.py",
    )
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        default=None,
        help="Local dataset shard (.tar.gz). Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compression_challenge_submission.zip"),
        help="Path to the submission zip to create.",
    )
    parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Disable the optional Rust bit-packing helper.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    shard_paths = args.shard if args.shard else default_eval_shards(repo_root)
    model_path = resolve_project_path(args.model, repo_root)
    output_path = resolve_project_path(args.output, repo_root)

    print("Step 1/4: loading model")
    model = TemporalMixtureModel.load(model_path)
    print("Step 2/4: counting records")
    record_count = count_shard_records(shard_paths, limit=args.limit)
    if record_count == 0:
        raise SystemExit("no records found in the provided shards")

    print("Step 3/4: evaluating predictor hit rate")
    hit_rate = model.topk_hit_rate(
        iter_shard_records(
            shard_paths,
            limit=args.limit,
            progress_desc="Loading records for hit-rate",
            progress_total=record_count,
        ),
        progress_desc="Scoring records",
        progress_total=record_count,
    )
    print("Step 4/4: encoding submission archive")
    payload = encode_records(
        iter_shard_records(
            shard_paths,
            limit=args.limit,
            progress_desc="Loading records for encoding",
            progress_total=record_count,
        ),
        model=model,
        use_rust=not args.no_rust,
        progress_desc="Encoding segments",
        progress_total=record_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("data.bin", payload)
        archive.write(model_path, arcname="model.npz")
        for relative_path in SUBMISSION_FILES:
            source = repo_root / relative_path
            archive.write(source, arcname=relative_path)

    print(f"records: {record_count}")
    print(f"top-8 hit rate: {hit_rate:.4f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
