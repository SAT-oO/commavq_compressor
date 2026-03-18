#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

from codec.dataset import count_shard_records, default_eval_shards, iter_shard_records
from codec.format import encode_records
from model.temporal import DEFAULT_LAGS, TemporalMixtureModel


ROOT = Path(__file__).resolve().parent
SUBMISSION_FILES = [
    "decompress.py",
    "codec/bits.py",
    "codec/format.py",
    "model/temporal.py",
    "runtime/entropy.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate full-dataset compression rate from a small balanced sample."
    )
    parser.add_argument(
        "--per-shard",
        type=int,
        default=32,
        help="Number of samples to use from each of the two evaluation shards.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAGS),
        help="Temporal lags to use in the temporary sample model.",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=8,
        help="Top-k shortlist size used only for reporting hit rate.",
    )
    parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Disable the optional Rust bit-packing helper.",
    )
    return parser.parse_args()


def build_sample_submission(
    records,
    model: TemporalMixtureModel,
    output_path: Path,
    use_rust: bool,
    progress_total: int | None = None,
) -> None:
    payload = encode_records(
        records,
        model=model,
        use_rust=use_rust,
        progress_desc="Encoding sample segments",
        progress_total=progress_total,
    )
    model_path = output_path.parent / "sample_model.npz"
    model.save(model_path)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("data.bin", payload)
        archive.write(model_path, arcname="model.npz")
        for relative_path in SUBMISSION_FILES:
            archive.write(ROOT / relative_path, arcname=relative_path)


def main() -> None:
    args = parse_args()
    shard_paths = default_eval_shards(ROOT)
    sample_target = count_shard_records(shard_paths, per_shard_limit=args.per_shard)
    print("Step 1/4: loading representative sample")
    sample_records = list(
        iter_shard_records(
            shard_paths,
            per_shard_limit=args.per_shard,
            progress_desc="Loading sample records",
            progress_total=sample_target,
        )
    )
    sample_count = len(sample_records)
    if sample_count == 0:
        raise SystemExit("no sample records found in the evaluation shards")

    print("Step 2/4: training sample model")
    model = TemporalMixtureModel.fit(
        sample_records,
        lag_steps=tuple(args.lags),
        progress_desc="Training sample records",
        progress_total=sample_count,
    )
    print("Step 3/4: measuring sample hit rate")
    hit_rate = model.topk_hit_rate(
        sample_records,
        top_k=args.report_top_k,
        progress_desc="Scoring sample records",
        progress_total=sample_count,
    )
    print("Step 4/4: building temporary sample submission")
    total_eval_records = count_shard_records(shard_paths)

    with tempfile.TemporaryDirectory(prefix="commavq_sample_") as tmp_dir:
        sample_zip = Path(tmp_dir) / "sample_submission.zip"
        build_sample_submission(
            sample_records,
            model=model,
            output_path=sample_zip,
            use_rust=not args.no_rust,
            progress_total=sample_count,
        )

        with zipfile.ZipFile(sample_zip) as archive:
            data_bin_info = archive.getinfo("data.bin")
            zip_size = sample_zip.stat().st_size
            fixed_overhead = zip_size - data_bin_info.compress_size
            bytes_per_record = data_bin_info.compress_size / sample_count

    source_bits_per_record = 1200 * 128 * 10 / 8
    sample_rate = (sample_count * source_bits_per_record) / zip_size
    estimated_full_zip_size = fixed_overhead + (bytes_per_record * total_eval_records)
    estimated_full_rate = (total_eval_records * source_bits_per_record) / estimated_full_zip_size

    print(f"sample records: {sample_count} ({args.per_shard} per shard target)")
    print(f"full eval records: {total_eval_records}")
    print(f"lags: {model.lag_steps.tolist()}")
    print(f"sample top-{args.report_top_k} hit rate: {hit_rate:.4f}")
    print(f"sample zip size: {zip_size / (1024 * 1024):.2f} MiB")
    print(f"sample observed compression rate: {sample_rate:.3f}x")
    print(f"estimated full zip size: {estimated_full_zip_size / (1024 * 1024):.2f} MiB")
    print(f"estimated full compression rate: {estimated_full_rate:.3f}x")
    print("note: this estimate is rough because the model is trained and measured on a small subset.")


if __name__ == "__main__":
    main()
