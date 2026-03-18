from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np


def _import_constriction():
    try:
        return importlib.import_module("constriction")
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "constriction"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return importlib.import_module("constriction")


class CategoricalFrameEncoder:
    def __init__(self):
        constriction = _import_constriction()
        self._model_family = constriction.stream.model.Categorical(perfect=False)
        self._encoder = constriction.stream.queue.RangeEncoder()

    def encode_frame(self, symbols: np.ndarray, probabilities: np.ndarray) -> None:
        self._encoder.encode(
            np.ascontiguousarray(np.asarray(symbols, dtype=np.int32)),
            self._model_family,
            np.ascontiguousarray(np.asarray(probabilities, dtype=np.float32)),
        )

    def to_bytes(self) -> bytes:
        compressed = self._encoder.get_compressed().astype("<u4", copy=False)
        return compressed.tobytes()


def encode_categorical_frames(
    symbols_per_frame: list[np.ndarray],
    probabilities_per_frame: list[np.ndarray],
) -> bytes:
    encoder = CategoricalFrameEncoder()
    for symbols, probabilities in zip(symbols_per_frame, probabilities_per_frame, strict=True):
        encoder.encode_frame(symbols, probabilities)
    return encoder.to_bytes()


class CategoricalFrameDecoder:
    def __init__(self, compressed_bytes: bytes):
        constriction = _import_constriction()
        self._model_family = constriction.stream.model.Categorical(perfect=False)

        if len(compressed_bytes) % 4 != 0:
            raise ValueError("compressed categorical payload must be divisible by 4 bytes")

        compressed_words = np.frombuffer(compressed_bytes, dtype="<u4").astype(np.uint32, copy=False)
        self._decoder = constriction.stream.queue.RangeDecoder(compressed_words)

    def decode_frame(self, probabilities: np.ndarray) -> np.ndarray:
        decoded = self._decoder.decode(
            self._model_family,
            np.ascontiguousarray(np.asarray(probabilities, dtype=np.float32)),
        )
        return np.asarray(decoded, dtype=np.int32)


def decode_categorical_frames(
    compressed_bytes: bytes,
    probabilities_per_frame: list[np.ndarray],
) -> list[np.ndarray]:
    decoder = CategoricalFrameDecoder(compressed_bytes)
    return [decoder.decode_frame(probabilities) for probabilities in probabilities_per_frame]
