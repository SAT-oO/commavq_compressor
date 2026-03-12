from __future__ import annotations

import json
import lzma
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .bits import pack_u16_10bit, unpack_u16_10bit
from .model import TransitionTopKModel


MAGIC = b"CVQTK001"


def encode_records(
    records: Iterable[tuple[str, np.ndarray]],
    model: TransitionTopKModel,
    use_rust: bool = True,
) -> bytes:
    sample_shape = None
    frames = None
    grid_shape = None
    positions = None
    payload_chunks: list[bytes] = []
    segments = []

    for name, tokens in records:
        token_array = np.asarray(tokens, dtype=np.uint16)
        if sample_shape is None:
            sample_shape = token_array.shape
            frames = sample_shape[0]
            grid_shape = sample_shape[1:]
            positions = int(np.prod(grid_shape))
            if positions != model.merged_topk.shape[0]:
                raise ValueError("model position count does not match token shape")
        elif token_array.shape != sample_shape:
            raise ValueError("all token arrays must have the same shape")

        flat_frames = token_array.reshape(frames, positions)
        warmup = flat_frames[: model.warmup_frames].reshape(-1)

        previous = flat_frames[model.warmup_frames - 1]
        rank_bytes = bytearray()
        escapes = []
        escape_count = 0
        for frame in flat_frames[model.warmup_frames :]:
            candidates = model.predict_topk(previous)
            matches = candidates == frame[:, None]
            hit_mask = matches.any(axis=1)
            ranks = np.where(hit_mask, matches.argmax(axis=1), model.top_k).astype(np.uint8)
            rank_bytes.extend(ranks.tobytes())

            if np.any(~hit_mask):
                misses = frame[~hit_mask]
                escapes.append(misses)
                escape_count += int(misses.size)
            previous = frame

        warmup_blob = lzma.compress(pack_u16_10bit(warmup, use_rust=use_rust), preset=9)
        rank_blob = lzma.compress(bytes(rank_bytes), preset=9)
        if escapes:
            escape_blob_raw = pack_u16_10bit(np.concatenate(escapes), use_rust=use_rust)
        else:
            escape_blob_raw = b""
        escape_blob = lzma.compress(escape_blob_raw, preset=9)

        payload_chunks.extend([warmup_blob, rank_blob, escape_blob])
        segments.append(
            {
                "name": name,
                "escape_count": escape_count,
                "warmup_len": len(warmup_blob),
                "rank_len": len(rank_blob),
                "escape_len": len(escape_blob),
            }
        )

    if sample_shape is None or frames is None or grid_shape is None or positions is None:
        raise ValueError("at least one record is required")

    manifest = {
        "frames": frames,
        "grid_shape": list(grid_shape),
        "positions": positions,
        "warmup_frames": model.warmup_frames,
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
    model: TransitionTopKModel,
    use_rust: bool = False,
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
    top_k = int(manifest["top_k"])

    if warmup_frames != model.warmup_frames or top_k != model.top_k:
        raise ValueError("payload metadata does not match loaded model")

    decoded: list[tuple[str, np.ndarray]] = []

    for segment in manifest["segments"]:
        warmup_blob = payload[offset : offset + int(segment["warmup_len"])]
        offset += int(segment["warmup_len"])
        rank_blob = payload[offset : offset + int(segment["rank_len"])]
        offset += int(segment["rank_len"])
        escape_blob = payload[offset : offset + int(segment["escape_len"])]
        offset += int(segment["escape_len"])

        segment_warmup = unpack_u16_10bit(
            lzma.decompress(warmup_blob),
            count=warmup_frames * positions,
            use_rust=use_rust,
        )
        ranks = np.frombuffer(lzma.decompress(rank_blob), dtype=np.uint8)
        segment_escape_count = int(segment["escape_count"])
        segment_escapes = unpack_u16_10bit(
            lzma.decompress(escape_blob),
            count=segment_escape_count,
            use_rust=use_rust,
        )

        flat_frames = np.empty((frames, positions), dtype=np.uint16)
        flat_frames[:warmup_frames] = segment_warmup.reshape(warmup_frames, positions)
        previous = flat_frames[warmup_frames - 1].copy()
        local_rank_offset = 0
        local_escape_offset = 0

        for frame_index in range(warmup_frames, frames):
            candidates = model.predict_topk(previous)
            frame_ranks = ranks[local_rank_offset : local_rank_offset + positions]
            local_rank_offset += positions

            frame = np.empty((positions,), dtype=np.uint16)
            hits = frame_ranks < top_k
            if np.any(hits):
                frame[hits] = candidates[np.arange(positions)[hits], frame_ranks[hits]]
            if np.any(~hits):
                miss_count = int((~hits).sum())
                frame[~hits] = segment_escapes[local_escape_offset : local_escape_offset + miss_count]
                local_escape_offset += miss_count

            flat_frames[frame_index] = frame
            previous = frame

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
