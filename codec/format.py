from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .bits import pack_u16_10bit, unpack_u16_10bit
from model.temporal import TemporalMixtureModel
from runtime.entropy import CategoricalFrameDecoder, CategoricalFrameEncoder


MAGIC = b"CVQTM001"


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
            if positions != model.mode_position_probs.shape[0]:
                raise ValueError("model position count does not match token shape")
        elif token_array.shape != sample_shape:
            raise ValueError("all token arrays must have the same shape")

        flat_frames = token_array.reshape(frames, positions)
        warmup = flat_frames[: model.warmup_frames].reshape(-1)

        mode_encoder = CategoricalFrameEncoder()
        novel_encoder = CategoricalFrameEncoder()
        for frame_index in range(model.warmup_frames, frames):
            context = flat_frames[:frame_index]
            mode_probs, novel_token_probs = model.predict(context)
            current_frame = flat_frames[frame_index]
            modes = model.copy_modes_for_frame(current_frame, context, model.lag_steps)
            mode_encoder.encode_frame(modes.astype(np.int32, copy=False), mode_probs)

            novel_mask = modes == model.novel_mode
            if np.any(novel_mask):
                novel_encoder.encode_frame(
                    current_frame[novel_mask].astype(np.int32, copy=False),
                    novel_token_probs[novel_mask],
                )

        warmup_blob = pack_u16_10bit(warmup, use_rust=use_rust)
        mode_blob = mode_encoder.to_bytes()
        novel_blob = novel_encoder.to_bytes()

        payload_chunks.extend([warmup_blob, mode_blob, novel_blob])
        segments.append(
            {
                "name": name,
                "warmup_len": len(warmup_blob),
                "mode_len": len(mode_blob),
                "novel_len": len(novel_blob),
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
        mode_blob = payload[offset : offset + int(segment["mode_len"])]
        offset += int(segment["mode_len"])
        novel_blob = payload[offset : offset + int(segment["novel_len"])]
        offset += int(segment["novel_len"])

        segment_warmup = unpack_u16_10bit(
            warmup_blob,
            count=warmup_frames * positions,
            use_rust=use_rust,
        )

        flat_frames = np.empty((frames, positions), dtype=np.uint16)
        flat_frames[:warmup_frames] = segment_warmup.reshape(warmup_frames, positions)
        mode_decoder = CategoricalFrameDecoder(mode_blob)
        novel_decoder = CategoricalFrameDecoder(novel_blob)
        for frame_index in range(warmup_frames, frames):
            context = flat_frames[:frame_index]
            mode_probs, novel_token_probs = model.predict(context)
            modes = mode_decoder.decode_frame(mode_probs).astype(np.int32, copy=False)

            frame = np.empty((positions,), dtype=np.uint16)
            for lag_index, lag in enumerate(model.lag_steps.tolist()):
                copy_mask = modes == lag_index
                if np.any(copy_mask):
                    frame[copy_mask] = flat_frames[frame_index - lag, copy_mask]

            novel_mask = modes == model.novel_mode
            if np.any(novel_mask):
                frame[novel_mask] = novel_decoder.decode_frame(novel_token_probs[novel_mask]).astype(
                    np.uint16, copy=False
                )

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
