"""
Range-coding wrappers using the `constriction` library.

Each FrameEncoder / FrameDecoder encodes / decodes one frame (128 tokens)
at a time, appending to / reading from a single binary stream per video segment.

API for batched (per-position) probability tables:
    encoder.encode(symbols, model_family, probs)   # probs: (S, V) float32
    decoder.decode(model_family, probs)            # returns int32 array (S,)
where model_family = constriction.stream.model.Categorical(perfect=False).
"""

import struct
import numpy as np
import constriction

# Shared Categorical model family object (stateless; reused across calls).
_CAT = constriction.stream.model.Categorical(perfect=False)

# Minimum probability floor to avoid -inf bits for unseen tokens.
_MIN_PROB = 1e-6


def _safe_probs(probs: np.ndarray) -> np.ndarray:
    """Clip and renormalise so every entry is positive and rows sum to 1."""
    p = np.clip(probs, _MIN_PROB, None).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    return p


class FrameEncoder:
    """Accumulates encoded frames into a single byte-stream."""

    def __init__(self) -> None:
        self._enc = constriction.stream.queue.RangeEncoder()
        self._frames = 0

    def encode_frame(self, tokens: np.ndarray, probs: np.ndarray) -> None:
        """
        tokens : (S,) int32 token IDs
        probs  : (S, VOCAB_SIZE) float32 probability table
        """
        p = _safe_probs(probs)
        self._enc.encode(tokens.astype(np.int32), _CAT, p)
        self._frames += 1

    def to_bytes(self) -> bytes:
        """Serialise compressed stream to bytes (little-endian uint32 words)."""
        arr = np.asarray(self._enc.get_compressed(), dtype="<u4")
        # Prepend a 4-byte frame count so the decoder knows how many to expect.
        header = struct.pack("<I", self._frames)
        return header + arr.tobytes()


class FrameDecoder:
    """Decodes frames from a byte-stream produced by FrameEncoder."""

    def __init__(self, data: bytes) -> None:
        (self._n_frames,) = struct.unpack_from("<I", data, 0)
        arr = np.frombuffer(data[4:], dtype="<u4").astype(np.uint32)
        self._dec = constriction.stream.queue.RangeDecoder(arr)
        self._decoded = 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def decode_frame(self, probs: np.ndarray) -> np.ndarray:
        """
        probs  : (S, VOCAB_SIZE) float32 probability table
        Returns: (S,) int32 token IDs
        """
        p = _safe_probs(probs)
        tokens = self._dec.decode(_CAT, p)
        self._decoded += 1
        return tokens
