#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from codec.format import decode_records, save_decoded_records
from model.temporal import TemporalMixtureModel


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get("OUTPUT_DIR", base_dir))
    print("Step 1/3: loading model")
    model = TemporalMixtureModel.load(base_dir / "model.npz")
    print("Step 2/3: reading compressed archive")
    payload = (base_dir / "data.bin").read_bytes()
    print("Step 3/3: decoding segments")
    records = decode_records(payload, model=model, use_rust=False, progress_desc="Decoding segments")
    save_decoded_records(records, output_dir)
    print(f"decoded {len(records)} segments into {output_dir}")


if __name__ == "__main__":
    main()
