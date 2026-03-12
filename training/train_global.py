#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codec.dataset import count_shard_records, default_eval_shards, iter_shard_records
from codec.model import TransitionTopKModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact top-k transition model for commaVQ.")
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
        help="Optional sample limit for faster iteration.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of candidate tokens to keep per context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/transition_topk_model.npz"),
        help="Output model path.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    shard_paths = args.shard if args.shard else default_eval_shards(ROOT)
    output_path = resolve_project_path(args.output)
    record_count = count_shard_records(shard_paths, limit=args.limit)
    if record_count == 0:
        raise SystemExit("no training records found")

    model = TransitionTopKModel.fit(
        iter_shard_records(shard_paths, limit=args.limit),
        top_k=args.top_k,
        warmup_frames=1,
    )
    hit_rate = model.topk_hit_rate(iter_shard_records(shard_paths, limit=args.limit))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"records: {record_count}")
    print(f"top-k: {model.top_k}")
    print(f"training hit rate: {hit_rate:.4f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
