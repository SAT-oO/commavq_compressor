#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from codec.format import decode_records, save_decoded_records
from codec.model import TransitionTopKModel


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(os.environ.get("OUTPUT_DIR", base_dir))
    model = TransitionTopKModel.load(base_dir / "model.npz")
    payload = (base_dir / "data.bin").read_bytes()
    records = decode_records(payload, model=model, use_rust=False)
    save_decoded_records(records, output_dir)
    print(f"decoded {len(records)} segments into {output_dir}")


if __name__ == "__main__":
    main()
