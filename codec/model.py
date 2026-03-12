from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


VOCAB_SIZE = 1024
POSITIONS = 128


@dataclasses.dataclass
class TransitionTopKModel:
    merged_topk: np.ndarray
    top_k: int
    warmup_frames: int = 1

    def __post_init__(self) -> None:
        self.merged_topk = np.asarray(self.merged_topk, dtype=np.uint16)
        expected_shape = (POSITIONS, VOCAB_SIZE, self.top_k)
        if self.merged_topk.shape != expected_shape:
            raise ValueError(f"expected merged_topk shape {expected_shape}, got {self.merged_topk.shape}")
        if self.warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")

    @classmethod
    def fit(
        cls,
        records,
        top_k: int = 8,
        warmup_frames: int = 1,
    ) -> "TransitionTopKModel":
        transition_counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
        position_counts = np.zeros((POSITIONS, VOCAB_SIZE), dtype=np.uint32)
        global_counts = np.zeros((VOCAB_SIZE,), dtype=np.uint64)

        for _, tokens in records:
            frames = tokens.reshape(tokens.shape[0], -1).astype(np.int64, copy=False)
            if frames.shape[1] != POSITIONS:
                raise ValueError(f"expected 128 positions, got {frames.shape[1]}")

            global_counts += np.bincount(
                frames.reshape(-1),
                minlength=VOCAB_SIZE,
            ).astype(np.uint64, copy=False)
            position_counts += np.apply_along_axis(
                lambda col: np.bincount(col, minlength=VOCAB_SIZE),
                0,
                frames,
            ).T.astype(np.uint32, copy=False)

            prev_tokens = frames[:-1].reshape(-1)
            next_tokens = frames[1:].reshape(-1)
            transition_counts += np.bincount(
                prev_tokens * VOCAB_SIZE + next_tokens,
                minlength=VOCAB_SIZE * VOCAB_SIZE,
            ).reshape(VOCAB_SIZE, VOCAB_SIZE).astype(np.uint32, copy=False)

        global_topk = _descending_topk(global_counts[None, :], top_k)[0]
        transition_topk = _descending_topk(transition_counts, top_k)
        position_topk = _descending_topk(position_counts, top_k)

        merged_topk = np.empty((POSITIONS, VOCAB_SIZE, top_k), dtype=np.uint16)
        for pos in range(POSITIONS):
            for prev_token in range(VOCAB_SIZE):
                merged_topk[pos, prev_token] = _merge_candidates(
                    prev_token=prev_token,
                    transition_row=transition_topk[prev_token],
                    position_row=position_topk[pos],
                    global_row=global_topk,
                    top_k=top_k,
                )

        return cls(merged_topk=merged_topk, top_k=top_k, warmup_frames=warmup_frames)

    def predict_topk(self, previous_frame_flat: np.ndarray) -> np.ndarray:
        previous_frame_flat = np.asarray(previous_frame_flat, dtype=np.int64).reshape(-1)
        if previous_frame_flat.shape[0] != POSITIONS:
            raise ValueError(f"expected previous frame with {POSITIONS} tokens")
        return self.merged_topk[np.arange(POSITIONS), previous_frame_flat]

    def topk_hit_rate(self, records) -> float:
        total = 0
        hits = 0
        for _, tokens in records:
            frames = tokens.reshape(tokens.shape[0], -1)
            previous = frames[self.warmup_frames - 1]
            for frame in frames[self.warmup_frames :]:
                candidates = self.predict_topk(previous)
                matches = candidates == frame[:, None]
                hits += int(matches.any(axis=1).sum())
                total += frame.shape[0]
                previous = frame
        return 0.0 if total == 0 else hits / total

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            kind=np.array(["transition_topk"]),
            merged_topk=self.merged_topk,
            top_k=np.array([self.top_k], dtype=np.int32),
            warmup_frames=np.array([self.warmup_frames], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TransitionTopKModel":
        payload = np.load(path, allow_pickle=False)
        kind = payload["kind"][0]
        if kind != "transition_topk":
            raise ValueError(f"unsupported model kind: {kind}")
        return cls(
            merged_topk=payload["merged_topk"],
            top_k=int(payload["top_k"][0]),
            warmup_frames=int(payload["warmup_frames"][0]),
        )


def _descending_topk(matrix: np.ndarray, top_k: int) -> np.ndarray:
    if matrix.shape[1] < top_k:
        raise ValueError("top_k exceeds vocabulary width")
    partition = np.argpartition(matrix, -top_k, axis=1)[:, -top_k:]
    partition_scores = np.take_along_axis(matrix, partition, axis=1)
    order = np.argsort(partition_scores, axis=1)[:, ::-1]
    return np.take_along_axis(partition, order, axis=1).astype(np.uint16, copy=False)


def _merge_candidates(
    prev_token: int,
    transition_row: np.ndarray,
    position_row: np.ndarray,
    global_row: np.ndarray,
    top_k: int,
) -> np.ndarray:
    merged = [prev_token]
    seen = {prev_token}

    for source in (transition_row, position_row, global_row):
        for value in np.asarray(source, dtype=np.int64).tolist():
            if value not in seen:
                seen.add(value)
                merged.append(value)
            if len(merged) == top_k:
                return np.asarray(merged, dtype=np.uint16)

    fill_value = global_row[0] if global_row.size else 0
    while len(merged) < top_k:
        merged.append(int(fill_value))
    return np.asarray(merged, dtype=np.uint16)
