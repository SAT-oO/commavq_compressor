from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


RUST_BINARY_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "target" / "release" / "video_compressor",
    Path(__file__).resolve().parents[1] / "target" / "debug" / "video_compressor",
)


def _find_rust_binary() -> Path | None:
    for candidate in RUST_BINARY_CANDIDATES:
        if candidate.exists():
            return candidate

    discovered = shutil.which("video_compressor")
    return Path(discovered) if discovered else None


def pack_u16_10bit(values: np.ndarray, use_rust: bool = True) -> bytes:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    if flat.size == 0:
        return b""
    if flat.max(initial=0) >= 1024:
        raise ValueError("token values must fit in 10 bits")

    rust_binary = _find_rust_binary() if use_rust else None
    raw = flat.astype("<u2", copy=False).tobytes()
    if rust_binary:
        completed = subprocess.run(
            [str(rust_binary), "pack10"],
            input=raw,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    out = bytearray()
    bit_buffer = 0
    bits_in_buffer = 0
    for value in flat.tolist():
        bit_buffer |= int(value) << bits_in_buffer
        bits_in_buffer += 10
        while bits_in_buffer >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer:
        out.append(bit_buffer & 0xFF)
    return bytes(out)


def unpack_u16_10bit(
    packed: bytes,
    count: int,
    use_rust: bool = True,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0,), dtype=np.uint16)

    rust_binary = _find_rust_binary() if use_rust else None
    if rust_binary:
        completed = subprocess.run(
            [str(rust_binary), "unpack10", str(count)],
            input=packed,
            check=True,
            capture_output=True,
        )
        return np.frombuffer(completed.stdout, dtype="<u2").astype(np.uint16, copy=False)

    out = np.empty(count, dtype=np.uint16)
    bit_buffer = 0
    bits_in_buffer = 0
    src_index = 0
    for idx in range(count):
        while bits_in_buffer < 10:
            if src_index >= len(packed):
                raise ValueError("not enough data to unpack requested token count")
            bit_buffer |= packed[src_index] << bits_in_buffer
            bits_in_buffer += 8
            src_index += 1
        out[idx] = bit_buffer & 0x3FF
        bit_buffer >>= 10
        bits_in_buffer -= 10
    return out
