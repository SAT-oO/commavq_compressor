"""
Range-coding wrappers using the `constriction` library.

This is the final-submission path: a single continuous range stream per sample,
with only a frame-count header.
"""

import struct

import constriction
import numpy as np

_CAT = constriction.stream.model.Categorical(perfect=False)
_MIN_PROB = 1e-6


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    """Clip and renormalise so every entry is positive and rows sum to 1."""
    p = np.round(probs, decimals=7)
    p = np.clip(p, _MIN_PROB, None).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    return p


class FrameEncoder:
    """Accumulates encoded frames into one continuous range stream."""

    def __init__(self) -> None:
        self._enc = constriction.stream.queue.RangeEncoder()
        self._frames = 0

    def encode_frame(self, tokens: np.ndarray, probs: np.ndarray) -> None:
        p = _safe_probs(probs)
        self._enc.encode(tokens.astype(np.int32), _CAT, p)
        self._frames += 1

    def to_bytes(self) -> bytes:
        arr = np.asarray(self._enc.get_compressed(), dtype="<u4")
        return struct.pack("<I", self._frames) + arr.tobytes()


class FrameDecoder:
    """Decodes frames from a continuous stream produced by FrameEncoder."""

    def __init__(self, data: bytes) -> None:
        (self._n_frames,) = struct.unpack_from("<I", data, 0)
        arr = np.frombuffer(data[4:], dtype="<u4").astype(np.uint32)
        self._dec = constriction.stream.queue.RangeDecoder(arr)
        self._decoded = 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def decode_frame(self, probs: np.ndarray) -> np.ndarray:
        if self._decoded >= self._n_frames:
            raise RuntimeError("Attempted to decode beyond available frames.")
        p = _safe_probs(probs)
        tokens = self._dec.decode(_CAT, p)
        self._decoded += 1
        return tokens
