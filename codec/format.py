from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .bits import pack_u16_10bit, unpack_u16_10bit
from model.temporal import TemporalMixtureModel
from runtime.entropy import CategoricalFrameDecoder, CategoricalFrameEncoder


MAGIC = b"CVQTM004"


def _progress(iterable, *, total: int | None, desc: str | None, unit: str):
    if desc is None:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


def encode_records(
    records: Iterable[tuple[str, np.ndarray]],
    model: TemporalMixtureModel,
    use_rust: bool = True,
    progress_desc: str | None = None,
    progress_total: int | None = None,
) -> bytes:
    sample_shape = None
    frames = None
    grid_shape = None
    positions = None
    payload_chunks: list[bytes] = []
    segments = []

    for name, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="segment"):
        token_array = np.asarray(tokens, dtype=np.uint16)
        if sample_shape is None:
            sample_shape = token_array.shape
            frames = sample_shape[0]
            grid_shape = sample_shape[1:]
            positions = int(np.prod(grid_shape))
            if positions != model.position_token_probs.shape[0]:
                raise ValueError("model position count does not match token shape")
        elif token_array.shape != sample_shape:
            raise ValueError("all token arrays must have the same shape")

        flat_frames = token_array.reshape(frames, positions)
        warmup = flat_frames[: model.warmup_frames].reshape(-1)

        entropy_encoder = CategoricalFrameEncoder()
        for frame_index in range(model.warmup_frames, frames):
            context = flat_frames[:frame_index]
            current_frame = flat_frames[frame_index]
            base_probs = model.effective_token_probabilities(context)
            for row_index in range(current_frame.size // 16):
                start = row_index * 16
                end = start + 16
                above_row = None if row_index == 0 else current_frame[start - 16 : start]
                row_probs = model.condition_row_probabilities(base_probs[start:end], above_row, row_index)
                entropy_encoder.encode_frame(current_frame[start:end].astype(np.int32, copy=False), row_probs)

        warmup_blob = pack_u16_10bit(warmup, use_rust=use_rust)
        entropy_blob = entropy_encoder.to_bytes()

        payload_chunks.extend([warmup_blob, entropy_blob])
        segments.append(
            {
                "name": name,
                "warmup_len": len(warmup_blob),
                "entropy_len": len(entropy_blob),
            }
        )

    if sample_shape is None or frames is None or grid_shape is None or positions is None:
        raise ValueError("at least one record is required")

    manifest = {
        "frames": frames,
        "grid_shape": list(grid_shape),
        "positions": positions,
        "warmup_frames": model.warmup_frames,
        "lag_steps": model.lag_steps.tolist(),
        "segments": segments,
    }
    manifest_blob = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    return b"".join(
        [
            MAGIC,
            struct.pack("<I", len(manifest_blob)),
            manifest_blob,
            *payload_chunks,
        ]
    )


def decode_records(
    payload: bytes,
    model: TemporalMixtureModel,
    use_rust: bool = False,
    progress_desc: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    if payload[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid archive magic")

    offset = len(MAGIC)
    manifest_len = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    manifest = json.loads(payload[offset : offset + manifest_len].decode("utf-8"))
    offset += manifest_len

    frames = int(manifest["frames"])
    grid_shape = tuple(manifest["grid_shape"])
    positions = int(manifest["positions"])
    warmup_frames = int(manifest["warmup_frames"])
    lag_steps = np.asarray(manifest["lag_steps"], dtype=np.int32)

    if warmup_frames != model.warmup_frames or not np.array_equal(lag_steps, model.lag_steps):
        raise ValueError("payload metadata does not match loaded model")

    decoded: list[tuple[str, np.ndarray]] = []

    for segment in _progress(
        manifest["segments"],
        total=len(manifest["segments"]),
        desc=progress_desc,
        unit="segment",
    ):
        warmup_blob = payload[offset : offset + int(segment["warmup_len"])]
        offset += int(segment["warmup_len"])
        entropy_blob = payload[offset : offset + int(segment["entropy_len"])]
        offset += int(segment["entropy_len"])

        segment_warmup = unpack_u16_10bit(
            warmup_blob,
            count=warmup_frames * positions,
            use_rust=use_rust,
        )

        flat_frames = np.empty((frames, positions), dtype=np.uint16)
        flat_frames[:warmup_frames] = segment_warmup.reshape(warmup_frames, positions)
        entropy_decoder = CategoricalFrameDecoder(entropy_blob)
        for frame_index in range(warmup_frames, frames):
            probabilities = model.effective_token_probabilities(flat_frames[:frame_index])
            frame = np.empty((positions,), dtype=np.uint16)
            for row_index in range(positions // 16):
                start = row_index * 16
                end = start + 16
                above_row = None if row_index == 0 else frame[start - 16 : start]
                row_probs = model.condition_row_probabilities(probabilities[start:end], above_row, row_index)
                frame[start:end] = entropy_decoder.decode_frame(row_probs).astype(np.uint16, copy=False)
            flat_frames[frame_index] = frame

        decoded.append((segment["name"], flat_frames.reshape((frames, *grid_shape)).astype(np.int16)))

    return decoded


def save_decoded_records(
    records: Iterable[tuple[str, np.ndarray]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, tokens in records:
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.save(handle, tokens, allow_pickle=False)
