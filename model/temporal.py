from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


VOCAB_SIZE = 1024
FRAME_ROWS = 8
FRAME_COLS = 16
POSITIONS = FRAME_ROWS * FRAME_COLS
DEFAULT_LAGS = (1,)
TOP_K = 32


def _progress(iterable, *, total: int | None, desc: str | None, unit: str):
    if desc is None:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


@dataclasses.dataclass
class TemporalMixtureModel:
    top_tokens: np.ndarray
    mode_probs: np.ndarray
    lag_steps: np.ndarray
    warmup_frames: int

    def __post_init__(self) -> None:
        self.top_tokens = np.asarray(self.top_tokens, dtype=np.uint16)
        self.mode_probs = np.asarray(self.mode_probs, dtype=np.float16)
        self.lag_steps = np.asarray(self.lag_steps, dtype=np.int32)

        if self.top_tokens.shape != (POSITIONS, VOCAB_SIZE, TOP_K):
            raise ValueError(f"expected top_tokens shape {(POSITIONS, VOCAB_SIZE, TOP_K)}, got {self.top_tokens.shape}")
        if self.mode_probs.shape != (TOP_K + 1,):
            raise ValueError(f"expected mode_probs shape {(TOP_K + 1,)}, got {self.mode_probs.shape}")
        if self.lag_steps.shape != (1,):
            raise ValueError("sparse lag predictor expects exactly one lag")
        if self.warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")

    @property
    def top_k(self) -> int:
        return TOP_K

    @property
    def primary_lag(self) -> int:
        return int(self.lag_steps[0])

    @classmethod
    def fit(
        cls,
        records,
        lag_steps: tuple[int, ...] = DEFAULT_LAGS,
        smoothing: float = 0.05,
        progress_desc: str | None = None,
        progress_total: int | None = None,
    ) -> "TemporalMixtureModel":
        lag_steps_arr = np.asarray(sorted(set(lag_steps)), dtype=np.int32)
        if lag_steps_arr.size == 0:
            raise ValueError("lag_steps must not be empty")

        primary_lag = int(lag_steps_arr[0])
        counts = np.zeros((POSITIONS, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
        position_indices = np.arange(POSITIONS, dtype=np.int32)

        for _, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[1] != POSITIONS:
                raise ValueError(f"expected 128 positions, got {frames.shape[1]}")
            if frames.shape[0] <= primary_lag:
                continue

            prev_frames = frames[:-primary_lag]
            next_frames = frames[primary_lag:]
            broadcast_positions = np.broadcast_to(position_indices, prev_frames.shape)
            np.add.at(
                counts,
                (broadcast_positions.reshape(-1), prev_frames.reshape(-1), next_frames.reshape(-1)),
                1,
            )

        flat_counts = counts.reshape(POSITIONS * VOCAB_SIZE, VOCAB_SIZE)
        top_tokens = np.argpartition(flat_counts, -TOP_K, axis=1)[:, -TOP_K:]
        top_counts = np.take_along_axis(flat_counts, top_tokens, axis=1).astype(np.float32, copy=False)
        order = np.argsort(top_counts, axis=1)[:, ::-1]
        top_tokens = np.take_along_axis(top_tokens, order, axis=1).astype(np.uint16, copy=False)
        top_counts = np.take_along_axis(top_counts, order, axis=1)

        mode_counts = np.zeros((TOP_K + 1,), dtype=np.uint64)
        top_tokens_shaped = top_tokens.reshape(POSITIONS, VOCAB_SIZE, TOP_K)
        for _, tokens in _progress(records, total=progress_total, desc=None, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[0] <= primary_lag:
                continue
            for frame_index in range(primary_lag, frames.shape[0]):
                previous = frames[frame_index - primary_lag]
                current = frames[frame_index]
                candidates = top_tokens_shaped[np.arange(POSITIONS), previous]
                matches = candidates == current[:, None]
                hit_mask = matches.any(axis=1)
                modes = np.full((POSITIONS,), TOP_K, dtype=np.int32)
                modes[hit_mask] = matches[hit_mask].argmax(axis=1).astype(np.int32, copy=False)
                mode_counts += np.bincount(modes, minlength=TOP_K + 1).astype(np.uint64, copy=False)

        mode_probs = (mode_counts.astype(np.float64) + smoothing) / (float(mode_counts.sum()) + (TOP_K + 1) * smoothing)

        return cls(
            top_tokens=top_tokens_shaped,
            mode_probs=mode_probs.astype(np.float16, copy=False),
            lag_steps=np.asarray([primary_lag], dtype=np.int32),
            warmup_frames=primary_lag,
        )

    def lookup_candidates(self, previous_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        previous = np.asarray(previous_frame, dtype=np.int64).reshape(POSITIONS)
        positions = np.arange(POSITIONS, dtype=np.int64)
        candidates = self.top_tokens[positions, previous]
        probs = np.broadcast_to(self.mode_probs.astype(np.float32, copy=False), (POSITIONS, TOP_K + 1))
        return candidates, probs.astype(np.float32, copy=False)

    def topk_hit_rate(
        self,
        records,
        top_k: int = 8,
        progress_desc: str | None = None,
        progress_total: int | None = None,
    ) -> float:
        shortlist = min(max(top_k, 1), TOP_K)
        total = 0
        hits = 0
        for _, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[0] <= self.warmup_frames:
                continue
            for frame_index in range(self.warmup_frames, frames.shape[0]):
                candidates, _ = self.lookup_candidates(frames[frame_index - self.primary_lag])
                current = frames[frame_index]
                hits += int((candidates[:, :shortlist] == current[:, None]).any(axis=1).sum())
                total += POSITIONS
        return 0.0 if total == 0 else hits / total

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            kind=np.array(["sparse_lag1_topk"]),
            top_tokens=self.top_tokens,
            mode_probs=self.mode_probs,
            lag_steps=self.lag_steps,
            warmup_frames=np.array([self.warmup_frames], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TemporalMixtureModel":
        payload = np.load(path, allow_pickle=False)
        kind = payload["kind"][0]
        if kind != "sparse_lag1_topk":
            raise ValueError(f"unsupported model kind: {kind}. retrain the model with the sparse lag codec.")
        return cls(
            top_tokens=payload["top_tokens"],
            mode_probs=payload["mode_probs"],
            lag_steps=payload["lag_steps"],
            warmup_frames=int(payload["warmup_frames"][0]),
        )
