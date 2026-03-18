from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .bits import pack_u16_10bit, unpack_u16_10bit
from model.temporal import FRAME_COLS, POSITIONS, TemporalMixtureModel
from runtime.entropy import CategoricalFrameDecoder, CategoricalFrameEncoder


MAGIC = b"CVQSP001"


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
            if positions != POSITIONS:
                raise ValueError("token shape does not match sparse lag codec geometry")
        elif token_array.shape != sample_shape:
            raise ValueError("all token arrays must have the same shape")

        flat_frames = token_array.reshape(frames, positions)
        warmup = flat_frames[: model.warmup_frames].reshape(-1)

        mode_encoder = CategoricalFrameEncoder()
        escaped_tokens: list[np.ndarray] = []
        escaped_count = 0
        for frame_index in range(model.warmup_frames, frames):
            previous_frame = flat_frames[frame_index - model.primary_lag]
            current_frame = flat_frames[frame_index]
            candidates, mode_probs = model.lookup_candidates(previous_frame)
            matches = candidates == current_frame[:, None]
            hit_mask = matches.any(axis=1)
            modes = np.full((positions,), model.top_k, dtype=np.int32)
            modes[hit_mask] = matches[hit_mask].argmax(axis=1).astype(np.int32, copy=False)
            mode_encoder.encode_frame(modes, mode_probs)

            if np.any(~hit_mask):
                escaped = current_frame[~hit_mask].astype(np.uint16, copy=False)
                escaped_tokens.append(escaped)
                escaped_count += escaped.size

        warmup_blob = pack_u16_10bit(warmup, use_rust=use_rust)
        mode_blob = mode_encoder.to_bytes()
        if escaped_tokens:
            escape_blob = pack_u16_10bit(np.concatenate(escaped_tokens), use_rust=use_rust)
        else:
            escape_blob = b""

        payload_chunks.extend([warmup_blob, mode_blob, escape_blob])
        segments.append(
            {
                "name": name,
                "warmup_len": len(warmup_blob),
                "mode_len": len(mode_blob),
                "escape_len": len(escape_blob),
                "escape_count": escaped_count,
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
        "top_k": model.top_k,
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
    top_k = int(manifest["top_k"])

    if positions != POSITIONS:
        raise ValueError("payload position count does not match sparse lag codec geometry")
    if warmup_frames != model.warmup_frames or not np.array_equal(lag_steps, model.lag_steps) or top_k != model.top_k:
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
        escape_blob = payload[offset : offset + int(segment["escape_len"])]
        offset += int(segment["escape_len"])

        segment_warmup = unpack_u16_10bit(
            warmup_blob,
            count=warmup_frames * positions,
            use_rust=use_rust,
        )
        escaped_tokens = unpack_u16_10bit(
            escape_blob,
            count=int(segment["escape_count"]),
            use_rust=use_rust,
        )

        flat_frames = np.empty((frames, positions), dtype=np.uint16)
        flat_frames[:warmup_frames] = segment_warmup.reshape(warmup_frames, positions)
        mode_decoder = CategoricalFrameDecoder(mode_blob)
        escape_offset = 0
        for frame_index in range(warmup_frames, frames):
            previous_frame = flat_frames[frame_index - model.primary_lag]
            candidates, mode_probs = model.lookup_candidates(previous_frame)
            modes = mode_decoder.decode_frame(mode_probs).astype(np.int32, copy=False)
            frame = np.empty((positions,), dtype=np.uint16)
            copied_mask = modes < model.top_k
            if np.any(copied_mask):
                frame[copied_mask] = candidates[copied_mask, modes[copied_mask]]
            escaped_mask = ~copied_mask
            if np.any(escaped_mask):
                escaped_count = int(escaped_mask.sum())
                frame[escaped_mask] = escaped_tokens[escape_offset : escape_offset + escaped_count]
                escape_offset += escaped_count
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
