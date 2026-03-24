#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np


def main() -> None:
    archive_path = Path(os.environ["PACKED_ARCHIVE"])
    unpacked_archive = Path(os.environ["UNPACKED_ARCHIVE"])
    gt_path = Path(os.environ.get("SMOKE_GT", "./test/smoke_gt.npy"))
    out_rel = Path(os.environ.get("SMOKE_OUT_REL", "smoke/token.npy"))

    out_path = unpacked_archive / out_rel
    if not out_path.exists():
        raise FileNotFoundError(f"Decoded smoke output missing: {out_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Smoke ground-truth file missing: {gt_path}")

    decoded = np.load(out_path)
    gt = np.load(gt_path)
    assert decoded.shape == gt.shape, (
        f"Shape mismatch: decoded {decoded.shape} vs gt {gt.shape}"
    )
    assert np.array_equal(decoded, gt), "Smoke decode does not match ground truth."

    # Compute effective compression ratio for this smoke sample.
    n_values = int(np.prod(gt.shape))
    raw_bytes = n_values * 10 / 8
    rate = raw_bytes / archive_path.stat().st_size

    print(f"Smoke evaluation passed: {out_rel}")
    print(f"Compression rate (smoke sample): {rate:.3f}x")


if __name__ == "__main__":
    main()
