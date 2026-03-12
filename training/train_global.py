#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codec.dataset import count_shard_records, default_eval_shards, iter_shard_records
from model.temporal import DEFAULT_LAGS, TemporalMixtureModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal mixture model for commaVQ.")
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
        "--lags",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAGS),
        help="Temporal lags to use as context frames.",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=8,
        help="Top-k shortlist size used only for reporting hit rate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/temporal_mixture_model.npz"),
        help="Output model path.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    shard_paths = args.shard if args.shard else default_eval_shards(ROOT)
    output_path = resolve_project_path(args.output)
    print("Step 1/3: counting training records")
    record_count = count_shard_records(shard_paths, limit=args.limit)
    if record_count == 0:
        raise SystemExit("no training records found")

    print("Step 2/3: fitting temporal model")
    model = TemporalMixtureModel.fit(
        iter_shard_records(
            shard_paths,
            limit=args.limit,
            progress_desc="Loading records for training",
            progress_total=record_count,
        ),
        lag_steps=tuple(args.lags),
        progress_desc="Training records",
        progress_total=record_count,
    )
    print("Step 3/3: measuring top-k hit rate")
    hit_rate = model.topk_hit_rate(
        iter_shard_records(
            shard_paths,
            limit=args.limit,
            progress_desc="Loading records for hit-rate",
            progress_total=record_count,
        ),
        top_k=args.report_top_k,
        progress_desc="Hit-rate records",
        progress_total=record_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"records: {record_count}")
    print(f"lags: {model.lag_steps.tolist()}")
    print(f"warmup frames: {model.warmup_frames}")
    print(f"top-{args.report_top_k} hit rate: {hit_rate:.4f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
