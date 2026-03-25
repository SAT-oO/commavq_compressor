"""
Range-coding wrappers using the `constriction` library.

Each FrameEncoder / FrameDecoder encodes / decodes one frame (128 tokens)
at a time, appending to / reading from a single binary stream per video segment.

API for batched (per-position) probability tables:
    encoder.encode(symbols, model_family, probs)   # probs: (S, V) float32
    decoder.decode(model_family, probs)            # returns int32 array (S,)
where model_family = constriction.stream.model.Categorical(perfect=False).
"""

import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import constriction

# Shared Categorical model family object (stateless; reused across calls).
_CAT = constriction.stream.model.Categorical(perfect=False)

# Minimum probability floor to avoid -inf bits for unseen tokens.
_MIN_PROB = 1e-6


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    """Clip and renormalise so every entry is positive and rows sum to 1."""
    # Round first to reduce tiny floating-point drift across encode/decode
    # environments before constriction quantises probabilities internally.
    p = np.round(probs, decimals=7)
    p = np.clip(p, _MIN_PROB, None).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    return p


def _rust_codec_bin() -> str:
    """Resolve Rust codec binary path for optional production backend."""
    return os.environ.get("RUST_CODEC_BIN", "target/release/video_compressor")


def _run_rust_encode_frame(tokens: np.ndarray, probs: np.ndarray) -> bytes:
    n_tokens, vocab = probs.shape
    header = struct.pack("<III", 1, n_tokens, vocab)
    body = (
        np.ascontiguousarray(probs.astype(np.float32)).tobytes()
        + np.ascontiguousarray(tokens.astype(np.int32)).tobytes()
    )
    proc = subprocess.run(
        [_rust_codec_bin(), "encode"],
        input=header + body,
        capture_output=True,
        check=True,
    )
    out = proc.stdout
    if len(out) < 8:
        raise RuntimeError("Rust encoder returned truncated output.")
    n_frames, comp_len = struct.unpack_from("<II", out, 0)
    if n_frames != 1:
        raise RuntimeError(f"Rust encoder returned unexpected frame count: {n_frames}")
    payload = out[8: 8 + comp_len]
    if len(payload) != comp_len:
        raise RuntimeError("Rust encoder payload length mismatch.")
    return payload


def _run_rust_decode_frame(payload: bytes, probs: np.ndarray) -> np.ndarray:
    n_tokens, vocab = probs.shape
    header = struct.pack("<III", 1, n_tokens, vocab)
    stream_header = struct.pack("<II", 1, len(payload))
    body = stream_header + payload + np.ascontiguousarray(probs.astype(np.float32)).tobytes()
    proc = subprocess.run(
        [_rust_codec_bin(), "decode"],
        input=header + body,
        capture_output=True,
        check=True,
    )
    out = proc.stdout
    expected = n_tokens * 4
    if len(out) != expected:
        raise RuntimeError(
            f"Rust decoder output size mismatch: expected {expected}, got {len(out)}"
        )
    return np.frombuffer(out, dtype="<i4").astype(np.int32, copy=False)


class FrameEncoder:
    """Accumulates independently encoded frame payloads."""

    def __init__(self) -> None:
        self._frame_payloads = []
        self._backend = os.environ.get("CODEC_BACKEND", "constriction").strip().lower()
        if self._backend not in {"constriction", "rust"}:
            raise ValueError("CODEC_BACKEND must be 'constriction' or 'rust'.")
        if self._backend == "rust" and not Path(_rust_codec_bin()).exists():
            raise FileNotFoundError(
                f"Rust codec binary not found: {_rust_codec_bin()}. "
                "Build with `cargo build --release` or set RUST_CODEC_BIN."
            )

    def encode_frame(self, tokens: np.ndarray, probs: np.ndarray) -> None:
        """
        tokens : (S,) int32 token IDs
        probs  : (S, VOCAB_SIZE) float32 probability table
        """
        p = _safe_probs(probs)
        if self._backend == "rust":
            self._frame_payloads.append(_run_rust_encode_frame(tokens, p))
            return

        enc = constriction.stream.queue.RangeEncoder()
        enc.encode(tokens.astype(np.int32), _CAT, p)
        arr = np.asarray(enc.get_compressed(), dtype="<u4")
        self._frame_payloads.append(arr.tobytes())

    def to_bytes(self) -> bytes:
        """Serialise framed payloads: [n_frames][len][payload]..."""
        parts = [struct.pack("<I", len(self._frame_payloads))]
        for payload in self._frame_payloads:
            parts.append(struct.pack("<I", len(payload)))
            parts.append(payload)
        return b"".join(parts)


class FrameDecoder:
    """Decodes framed payloads produced by FrameEncoder."""

    def __init__(self, data: bytes) -> None:
        self._backend = os.environ.get("CODEC_BACKEND", "constriction").strip().lower()
        if self._backend not in {"constriction", "rust"}:
            raise ValueError("CODEC_BACKEND must be 'constriction' or 'rust'.")
        if self._backend == "rust" and not Path(_rust_codec_bin()).exists():
            raise FileNotFoundError(
                f"Rust codec binary not found: {_rust_codec_bin()}. "
                "Build with `cargo build --release` or set RUST_CODEC_BIN."
            )

        (self._n_frames,) = struct.unpack_from("<I", data, 0)
        self._payloads = []
        offset = 4
        for _ in range(self._n_frames):
            (n_bytes,) = struct.unpack_from("<I", data, offset)
            offset += 4
            payload = data[offset: offset + n_bytes]
            offset += n_bytes
            self._payloads.append(payload)
        self._decoded = 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def decode_frame(self, probs: np.ndarray) -> np.ndarray:
        """
        probs  : (S, VOCAB_SIZE) float32 probability table
        Returns: (S,) int32 token IDs
        """
        if self._decoded >= self._n_frames:
            raise RuntimeError("Attempted to decode beyond available frames.")

        p = _safe_probs(probs)
        if self._backend == "rust":
            tokens = _run_rust_decode_frame(self._payloads[self._decoded], p)
            self._decoded += 1
            return tokens

        arr = np.frombuffer(self._payloads[self._decoded], dtype="<u4").astype(np.uint32)
        dec = constriction.stream.queue.RangeDecoder(arr)
        tokens = dec.decode(_CAT, p)
        self._decoded += 1
        return tokens
