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
    # Round first to reduce tiny floating-point drift across encode/decode
    # environments before constriction quantises probabilities internally.
    p = np.round(probs, decimals=7)
    p = np.clip(p, _MIN_PROB, None).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    return p


class FrameEncoder:
    """Accumulates independently encoded frame payloads."""

    def __init__(self) -> None:
        self._frame_payloads = []

    def encode_frame(self, tokens: np.ndarray, probs: np.ndarray) -> None:
        """
        tokens : (S,) int32 token IDs
        probs  : (S, VOCAB_SIZE) float32 probability table
        """
        p = _safe_probs(probs)
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
        arr = np.frombuffer(self._payloads[self._decoded], dtype="<u4").astype(np.uint32)
        dec = constriction.stream.queue.RangeDecoder(arr)
        tokens = dec.decode(_CAT, p)
        self._decoded += 1
        return tokens
