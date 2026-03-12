from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


VOCAB_SIZE = 1024
POSITIONS = 128
DEFAULT_LAGS = (1, 2, 4, 8)


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
    global_probs: np.ndarray
    position_probs: np.ndarray
    lag_transition_probs: np.ndarray
    lag_steps: np.ndarray
    component_weights: np.ndarray
    warmup_frames: int

    def __post_init__(self) -> None:
        self.global_probs = np.asarray(self.global_probs, dtype=np.float16)
        self.position_probs = np.asarray(self.position_probs, dtype=np.float16)
        self.lag_transition_probs = np.asarray(self.lag_transition_probs, dtype=np.float16)
        self.lag_steps = np.asarray(self.lag_steps, dtype=np.int32)
        self.component_weights = np.asarray(self.component_weights, dtype=np.float32)

        if self.global_probs.shape != (VOCAB_SIZE,):
            raise ValueError(f"expected global_probs shape {(VOCAB_SIZE,)}, got {self.global_probs.shape}")
        if self.position_probs.shape != (POSITIONS, VOCAB_SIZE):
            raise ValueError(
                f"expected position_probs shape {(POSITIONS, VOCAB_SIZE)}, got {self.position_probs.shape}"
            )
        if self.lag_transition_probs.shape[1:] != (VOCAB_SIZE, VOCAB_SIZE):
            raise ValueError(
                f"expected lag_transition_probs trailing shape {(VOCAB_SIZE, VOCAB_SIZE)}, "
                f"got {self.lag_transition_probs.shape[1:]}"
            )
        if self.lag_transition_probs.shape[0] != self.lag_steps.shape[0]:
            raise ValueError("lag_transition_probs and lag_steps length mismatch")
        if self.component_weights.shape != (self.lag_steps.shape[0] + 2,):
            raise ValueError("component_weights must contain [global, position, per-lag] weights")
        if self.warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")

    @classmethod
    def fit(
        cls,
        records,
        lag_steps: tuple[int, ...] = DEFAULT_LAGS,
        smoothing: float = 0.25,
        progress_desc: str | None = None,
        progress_total: int | None = None,
    ) -> "TemporalMixtureModel":
        lag_steps_arr = np.asarray(sorted(set(lag_steps)), dtype=np.int32)
        if lag_steps_arr.size == 0:
            raise ValueError("lag_steps must not be empty")

        global_counts = np.zeros((VOCAB_SIZE,), dtype=np.uint64)
        position_counts = np.zeros((POSITIONS, VOCAB_SIZE), dtype=np.uint32)
        lag_transition_counts = np.zeros((lag_steps_arr.size, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
        lag_match_counts = np.zeros((lag_steps_arr.size,), dtype=np.uint64)
        lag_total_counts = np.zeros((lag_steps_arr.size,), dtype=np.uint64)

        for _, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[1] != POSITIONS:
                raise ValueError(f"expected 128 positions, got {frames.shape[1]}")

            global_counts += np.bincount(frames.reshape(-1), minlength=VOCAB_SIZE).astype(np.uint64, copy=False)
            for pos in range(POSITIONS):
                position_counts[pos] += np.bincount(frames[:, pos], minlength=VOCAB_SIZE).astype(
                    np.uint32, copy=False
                )

            for lag_index, lag in enumerate(lag_steps_arr):
                if frames.shape[0] <= lag:
                    continue
                prev_tokens = frames[:-lag].reshape(-1)
                next_tokens = frames[lag:].reshape(-1)
                lag_transition_counts[lag_index] += np.bincount(
                    prev_tokens * VOCAB_SIZE + next_tokens,
                    minlength=VOCAB_SIZE * VOCAB_SIZE,
                ).reshape(VOCAB_SIZE, VOCAB_SIZE).astype(np.uint32, copy=False)
                lag_match_counts[lag_index] += int((prev_tokens == next_tokens).sum())
                lag_total_counts[lag_index] += next_tokens.size

        global_probs = _normalize_counts(global_counts, smoothing)
        position_probs = _normalize_rows(position_counts, smoothing)
        lag_transition_probs = _normalize_rows(lag_transition_counts, smoothing)
        component_weights = _derive_component_weights(lag_steps_arr, lag_match_counts, lag_total_counts)

        return cls(
            global_probs=global_probs.astype(np.float16, copy=False),
            position_probs=position_probs.astype(np.float16, copy=False),
            lag_transition_probs=lag_transition_probs.astype(np.float16, copy=False),
            lag_steps=lag_steps_arr,
            component_weights=component_weights.astype(np.float32, copy=False),
            warmup_frames=int(lag_steps_arr.max()),
        )

    def predict_probabilities(self, context_frames: np.ndarray) -> np.ndarray:
        context_frames = np.asarray(context_frames, dtype=np.int64)
        if context_frames.ndim != 2 or context_frames.shape[1] != POSITIONS:
            raise ValueError(f"expected context_frames shape [T, {POSITIONS}]")
        if context_frames.shape[0] < self.warmup_frames:
            raise ValueError(
                f"expected at least {self.warmup_frames} context frames, got {context_frames.shape[0]}"
            )

        probs = (
            float(self.component_weights[0]) * self.global_probs.astype(np.float32, copy=False)[None, :]
            + float(self.component_weights[1]) * self.position_probs.astype(np.float32, copy=False)
        )

        lag_tables = self.lag_transition_probs.astype(np.float32, copy=False)
        for lag_index, lag in enumerate(self.lag_steps.tolist()):
            prev_tokens = context_frames[-lag].astype(np.int64, copy=False)
            probs += float(self.component_weights[lag_index + 2]) * lag_tables[lag_index, prev_tokens]

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= probs_sum
        return probs.astype(np.float32, copy=False)

    def topk_hit_rate(
        self,
        records,
        top_k: int = 8,
        progress_desc: str | None = None,
        progress_total: int | None = None,
    ) -> float:
        total = 0
        hits = 0
        for _, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[0] <= self.warmup_frames:
                continue
            for frame_index in range(self.warmup_frames, frames.shape[0]):
                probabilities = self.predict_probabilities(frames[:frame_index])
                topk = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
                hits += int((topk == frames[frame_index][:, None]).any(axis=1).sum())
                total += POSITIONS
        return 0.0 if total == 0 else hits / total

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            kind=np.array(["temporal_mixture"]),
            global_probs=self.global_probs,
            position_probs=self.position_probs,
            lag_transition_probs=self.lag_transition_probs,
            lag_steps=self.lag_steps,
            component_weights=self.component_weights.astype(np.float16, copy=False),
            warmup_frames=np.array([self.warmup_frames], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TemporalMixtureModel":
        payload = np.load(path, allow_pickle=False)
        kind = payload["kind"][0]
        if kind != "temporal_mixture":
            raise ValueError(f"unsupported model kind: {kind}")
        return cls(
            global_probs=payload["global_probs"],
            position_probs=payload["position_probs"],
            lag_transition_probs=payload["lag_transition_probs"],
            lag_steps=payload["lag_steps"],
            component_weights=payload["component_weights"].astype(np.float32, copy=False),
            warmup_frames=int(payload["warmup_frames"][0]),
        )


def _normalize_counts(counts: np.ndarray, smoothing: float) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    numerators = counts + smoothing
    return numerators / numerators.sum()


def _normalize_rows(counts: np.ndarray, smoothing: float) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    numerators = counts + smoothing
    denominators = numerators.sum(axis=-1, keepdims=True)
    return numerators / denominators


def _derive_component_weights(
    lag_steps: np.ndarray,
    lag_match_counts: np.ndarray,
    lag_total_counts: np.ndarray,
) -> np.ndarray:
    lag_copy_rates = np.divide(
        lag_match_counts.astype(np.float64),
        np.maximum(lag_total_counts, 1),
        out=np.zeros_like(lag_match_counts, dtype=np.float64),
        where=lag_total_counts > 0,
    )
    lag_scores = lag_copy_rates / np.sqrt(lag_steps.astype(np.float64))
    raw_weights = np.concatenate(
        [
            np.array([0.06, 0.18], dtype=np.float64),  # global and position priors
            np.maximum(lag_scores, 1e-3),
        ]
    )
    return (raw_weights / raw_weights.sum()).astype(np.float32)
