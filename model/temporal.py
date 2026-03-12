from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


VOCAB_SIZE = 1024
FRAME_ROWS = 8
FRAME_COLS = 16
POSITIONS = FRAME_ROWS * FRAME_COLS
DEFAULT_LAGS = (1, 2, 4, 8)
CLUSTER_COUNT = 4
COPY_RATE_THRESHOLDS = (0.38, 0.24)


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
    mode_global_probs: np.ndarray
    mode_position_probs: np.ndarray
    mode_cluster_probs: np.ndarray
    global_token_probs: np.ndarray
    row_token_probs: np.ndarray
    position_token_probs: np.ndarray
    cluster_token_probs: np.ndarray
    cluster_position_token_probs: np.ndarray
    lag_transition_probs: np.ndarray
    row_lag_transition_probs: np.ndarray
    lag_steps: np.ndarray
    token_component_weights: np.ndarray
    mode_component_weights: np.ndarray
    warmup_frames: int

    def __post_init__(self) -> None:
        self.mode_global_probs = np.asarray(self.mode_global_probs, dtype=np.float16)
        self.mode_position_probs = np.asarray(self.mode_position_probs, dtype=np.float16)
        self.mode_cluster_probs = np.asarray(self.mode_cluster_probs, dtype=np.float16)
        self.global_token_probs = np.asarray(self.global_token_probs, dtype=np.float16)
        self.row_token_probs = np.asarray(self.row_token_probs, dtype=np.float16)
        self.position_token_probs = np.asarray(self.position_token_probs, dtype=np.float16)
        self.cluster_token_probs = np.asarray(self.cluster_token_probs, dtype=np.float16)
        self.cluster_position_token_probs = np.asarray(self.cluster_position_token_probs, dtype=np.float16)
        self.lag_transition_probs = np.asarray(self.lag_transition_probs, dtype=np.float16)
        self.row_lag_transition_probs = np.asarray(self.row_lag_transition_probs, dtype=np.float16)
        self.lag_steps = np.asarray(self.lag_steps, dtype=np.int32)
        self.token_component_weights = np.asarray(self.token_component_weights, dtype=np.float32)
        self.mode_component_weights = np.asarray(self.mode_component_weights, dtype=np.float32)

        mode_count = self.lag_steps.size + 1
        if self.mode_global_probs.shape != (mode_count,):
            raise ValueError(f"expected mode_global_probs shape {(mode_count,)}, got {self.mode_global_probs.shape}")
        if self.mode_position_probs.shape != (POSITIONS, mode_count):
            raise ValueError(
                f"expected mode_position_probs shape {(POSITIONS, mode_count)}, got {self.mode_position_probs.shape}"
            )
        if self.mode_cluster_probs.shape != (CLUSTER_COUNT, mode_count):
            raise ValueError(
                f"expected mode_cluster_probs shape {(CLUSTER_COUNT, mode_count)}, got {self.mode_cluster_probs.shape}"
            )
        if self.global_token_probs.shape != (VOCAB_SIZE,):
            raise ValueError(f"expected global_token_probs shape {(VOCAB_SIZE,)}, got {self.global_token_probs.shape}")
        if self.row_token_probs.shape != (FRAME_ROWS, VOCAB_SIZE):
            raise ValueError(
                f"expected row_token_probs shape {(FRAME_ROWS, VOCAB_SIZE)}, got {self.row_token_probs.shape}"
            )
        if self.position_token_probs.shape != (POSITIONS, VOCAB_SIZE):
            raise ValueError(
                f"expected position_token_probs shape {(POSITIONS, VOCAB_SIZE)}, got {self.position_token_probs.shape}"
            )
        if self.cluster_token_probs.shape != (CLUSTER_COUNT, VOCAB_SIZE):
            raise ValueError(
                f"expected cluster_token_probs shape {(CLUSTER_COUNT, VOCAB_SIZE)}, got {self.cluster_token_probs.shape}"
            )
        if self.cluster_position_token_probs.shape != (CLUSTER_COUNT, POSITIONS, VOCAB_SIZE):
            raise ValueError(
                "expected cluster_position_token_probs shape "
                f"{(CLUSTER_COUNT, POSITIONS, VOCAB_SIZE)}, got {self.cluster_position_token_probs.shape}"
            )
        if self.lag_transition_probs.shape != (self.lag_steps.size, VOCAB_SIZE, VOCAB_SIZE):
            raise ValueError(
                "expected lag_transition_probs shape "
                f"{(self.lag_steps.size, VOCAB_SIZE, VOCAB_SIZE)}, got {self.lag_transition_probs.shape}"
            )
        if self.row_lag_transition_probs.shape != (self.lag_steps.size, FRAME_ROWS, VOCAB_SIZE, VOCAB_SIZE):
            raise ValueError(
                "expected row_lag_transition_probs shape "
                f"{(self.lag_steps.size, FRAME_ROWS, VOCAB_SIZE, VOCAB_SIZE)}, "
                f"got {self.row_lag_transition_probs.shape}"
            )
        if self.warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")

    @property
    def novel_mode(self) -> int:
        return int(self.lag_steps.size)

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

        warmup_frames = int(lag_steps_arr.max())
        mode_count = lag_steps_arr.size + 1
        position_indices = np.arange(POSITIONS, dtype=np.int32)
        row_indices = position_indices // FRAME_COLS

        mode_global_counts = np.zeros((mode_count,), dtype=np.uint64)
        mode_position_counts = np.zeros((POSITIONS, mode_count), dtype=np.uint32)
        mode_cluster_counts = np.zeros((CLUSTER_COUNT, mode_count), dtype=np.uint32)

        global_token_counts = np.zeros((VOCAB_SIZE,), dtype=np.uint64)
        row_token_counts = np.zeros((FRAME_ROWS, VOCAB_SIZE), dtype=np.uint32)
        position_token_counts = np.zeros((POSITIONS, VOCAB_SIZE), dtype=np.uint32)
        cluster_token_counts = np.zeros((CLUSTER_COUNT, VOCAB_SIZE), dtype=np.uint32)
        cluster_position_token_counts = np.zeros((CLUSTER_COUNT, POSITIONS, VOCAB_SIZE), dtype=np.uint32)
        lag_transition_counts = np.zeros((lag_steps_arr.size, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
        row_lag_transition_counts = np.zeros((lag_steps_arr.size, FRAME_ROWS, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)

        for _, tokens in _progress(records, total=progress_total, desc=progress_desc, unit="record"):
            frames = np.asarray(tokens, dtype=np.int64).reshape(tokens.shape[0], -1)
            if frames.shape[1] != POSITIONS:
                raise ValueError(f"expected 128 positions, got {frames.shape[1]}")
            if frames.shape[0] <= warmup_frames:
                continue

            for frame_index in range(warmup_frames, frames.shape[0]):
                context = frames[:frame_index]
                current = frames[frame_index]
                cluster = _context_cluster(context)
                modes = cls.copy_modes_for_frame(current, context, lag_steps_arr)

                mode_global_counts += np.bincount(modes, minlength=mode_count).astype(np.uint64, copy=False)
                np.add.at(mode_position_counts, (position_indices, modes), 1)
                mode_cluster_counts[cluster] += np.bincount(modes, minlength=mode_count).astype(np.uint32, copy=False)

                novel_mask = modes == (mode_count - 1)
                if not np.any(novel_mask):
                    continue

                novel_tokens = current[novel_mask]
                novel_positions = position_indices[novel_mask]
                novel_rows = row_indices[novel_mask]

                global_token_counts += np.bincount(novel_tokens, minlength=VOCAB_SIZE).astype(np.uint64, copy=False)
                cluster_token_counts[cluster] += np.bincount(novel_tokens, minlength=VOCAB_SIZE).astype(
                    np.uint32, copy=False
                )
                np.add.at(position_token_counts, (novel_positions, novel_tokens), 1)
                np.add.at(row_token_counts, (novel_rows, novel_tokens), 1)
                np.add.at(
                    cluster_position_token_counts,
                    (np.full(novel_positions.size, cluster, dtype=np.int32), novel_positions, novel_tokens),
                    1,
                )

                for lag_index, lag in enumerate(lag_steps_arr):
                    prev_tokens = context[-lag][novel_mask]
                    np.add.at(lag_transition_counts[lag_index], (prev_tokens, novel_tokens), 1)
                    np.add.at(row_lag_transition_counts[lag_index], (novel_rows, prev_tokens, novel_tokens), 1)

        return cls(
            mode_global_probs=_normalize_counts(mode_global_counts, smoothing).astype(np.float16, copy=False),
            mode_position_probs=_normalize_rows(mode_position_counts, smoothing).astype(np.float16, copy=False),
            mode_cluster_probs=_normalize_rows(mode_cluster_counts, smoothing).astype(np.float16, copy=False),
            global_token_probs=_normalize_counts(global_token_counts, smoothing).astype(np.float16, copy=False),
            row_token_probs=_normalize_rows(row_token_counts, smoothing).astype(np.float16, copy=False),
            position_token_probs=_normalize_rows(position_token_counts, smoothing).astype(np.float16, copy=False),
            cluster_token_probs=_normalize_rows(cluster_token_counts, smoothing).astype(np.float16, copy=False),
            cluster_position_token_probs=_normalize_rows(cluster_position_token_counts, smoothing).astype(
                np.float16, copy=False
            ),
            lag_transition_probs=_normalize_rows(lag_transition_counts, smoothing).astype(np.float16, copy=False),
            row_lag_transition_probs=_normalize_rows(row_lag_transition_counts, smoothing).astype(
                np.float16, copy=False
            ),
            lag_steps=lag_steps_arr,
            token_component_weights=_derive_token_weights(lag_steps_arr),
            mode_component_weights=np.asarray([0.08, 0.42, 0.50], dtype=np.float32),
            warmup_frames=warmup_frames,
        )

    @staticmethod
    def copy_modes_for_frame(current_frame: np.ndarray, context_frames: np.ndarray, lag_steps: np.ndarray) -> np.ndarray:
        modes = np.full(POSITIONS, lag_steps.size, dtype=np.int32)
        assigned = np.zeros(POSITIONS, dtype=bool)
        for lag_index, lag in enumerate(lag_steps.tolist()):
            matches = (current_frame == context_frames[-lag]) & (~assigned)
            modes[matches] = lag_index
            assigned |= matches
        return modes

    def predict_mode_probabilities(self, context_frames: np.ndarray) -> np.ndarray:
        context_frames = np.asarray(context_frames, dtype=np.int64)
        if context_frames.ndim != 2 or context_frames.shape[1] != POSITIONS:
            raise ValueError(f"expected context_frames shape [T, {POSITIONS}]")
        if context_frames.shape[0] < self.warmup_frames:
            raise ValueError(
                f"expected at least {self.warmup_frames} context frames, got {context_frames.shape[0]}"
            )

        cluster = _context_cluster(context_frames)
        probs = (
            float(self.mode_component_weights[0]) * self.mode_global_probs.astype(np.float32, copy=False)[None, :]
            + float(self.mode_component_weights[1]) * self.mode_position_probs.astype(np.float32, copy=False)
            + float(self.mode_component_weights[2]) * self.mode_cluster_probs.astype(np.float32, copy=False)[cluster][
                None, :
            ]
        )

        dynamic = np.zeros((POSITIONS, self.novel_mode + 1), dtype=np.float32)
        max_copy_signal = np.zeros((POSITIONS,), dtype=np.float32)
        for lag_index, lag in enumerate(self.lag_steps.tolist()):
            anchor = context_frames[-lag]
            prev_anchor_index = context_frames.shape[0] - lag - 1
            if prev_anchor_index >= 0:
                signal = (anchor == context_frames[prev_anchor_index]).astype(np.float32)
            else:
                signal = np.zeros((POSITIONS,), dtype=np.float32)
            dynamic[:, lag_index] = signal
            max_copy_signal = np.maximum(max_copy_signal, signal)
        dynamic[:, self.novel_mode] = 1.0 - max_copy_signal
        probs += 0.15 * dynamic

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= probs_sum
        return probs.astype(np.float32, copy=False)

    def predict_novel_token_probabilities(self, context_frames: np.ndarray) -> np.ndarray:
        context_frames = np.asarray(context_frames, dtype=np.int64)
        if context_frames.ndim != 2 or context_frames.shape[1] != POSITIONS:
            raise ValueError(f"expected context_frames shape [T, {POSITIONS}]")
        if context_frames.shape[0] < self.warmup_frames:
            raise ValueError(
                f"expected at least {self.warmup_frames} context frames, got {context_frames.shape[0]}"
            )

        cluster = _context_cluster(context_frames)
        rows = np.arange(POSITIONS, dtype=np.int32) // FRAME_COLS

        base_count = 5
        lag_count = self.lag_steps.size
        weights = self.token_component_weights
        probs = (
            float(weights[0]) * self.global_token_probs.astype(np.float32, copy=False)[None, :]
            + float(weights[1]) * self.position_token_probs.astype(np.float32, copy=False)
            + float(weights[2]) * self.row_token_probs.astype(np.float32, copy=False)[rows]
            + float(weights[3]) * self.cluster_token_probs.astype(np.float32, copy=False)[cluster][None, :]
            + float(weights[4])
            * self.cluster_position_token_probs.astype(np.float32, copy=False)[cluster]
        )

        lag_transition_probs = self.lag_transition_probs.astype(np.float32, copy=False)
        row_lag_transition_probs = self.row_lag_transition_probs.astype(np.float32, copy=False)
        shared_weights = weights[base_count : base_count + lag_count]
        row_weights = weights[base_count + lag_count :]

        for lag_index, lag in enumerate(self.lag_steps.tolist()):
            prev_tokens = context_frames[-lag]
            probs += float(shared_weights[lag_index]) * lag_transition_probs[lag_index, prev_tokens]
            probs += float(row_weights[lag_index]) * row_lag_transition_probs[lag_index, rows, prev_tokens]

        copy_token_indices = np.stack([context_frames[-lag] for lag in self.lag_steps.tolist()], axis=0)
        masked_probs = probs.copy()
        for lag_tokens in copy_token_indices:
            masked_probs[np.arange(POSITIONS), lag_tokens] = 1e-8

        row_sums = masked_probs.sum(axis=1, keepdims=True)
        fallback = (
            0.30 * self.position_token_probs.astype(np.float32, copy=False)
            + 0.20 * self.row_token_probs.astype(np.float32, copy=False)[rows]
            + 0.50 * self.global_token_probs.astype(np.float32, copy=False)[None, :]
        )
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            masked_probs[zero_rows] = fallback[zero_rows]
            row_sums = masked_probs.sum(axis=1, keepdims=True)

        masked_probs /= row_sums
        return masked_probs.astype(np.float32, copy=False)

    def predict(self, context_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_mode_probabilities(context_frames), self.predict_novel_token_probabilities(context_frames)

    def effective_token_probabilities(self, context_frames: np.ndarray) -> np.ndarray:
        mode_probs, novel_token_probs = self.predict(context_frames)
        effective = mode_probs[:, self.novel_mode][:, None] * novel_token_probs
        for lag_index, lag in enumerate(self.lag_steps.tolist()):
            effective[np.arange(POSITIONS), context_frames[-lag]] += mode_probs[:, lag_index]
        return effective

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
                probabilities = self.effective_token_probabilities(frames[:frame_index])
                topk = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
                hits += int((topk == frames[frame_index][:, None]).any(axis=1).sum())
                total += POSITIONS
        return 0.0 if total == 0 else hits / total

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            kind=np.array(["temporal_mixture"]),
            mode_global_probs=self.mode_global_probs,
            mode_position_probs=self.mode_position_probs,
            mode_cluster_probs=self.mode_cluster_probs,
            global_token_probs=self.global_token_probs,
            row_token_probs=self.row_token_probs,
            position_token_probs=self.position_token_probs,
            cluster_token_probs=self.cluster_token_probs,
            cluster_position_token_probs=self.cluster_position_token_probs,
            lag_transition_probs=self.lag_transition_probs,
            row_lag_transition_probs=self.row_lag_transition_probs,
            lag_steps=self.lag_steps,
            token_component_weights=self.token_component_weights.astype(np.float16, copy=False),
            mode_component_weights=self.mode_component_weights.astype(np.float16, copy=False),
            warmup_frames=np.array([self.warmup_frames], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "TemporalMixtureModel":
        payload = np.load(path, allow_pickle=False)
        kind = payload["kind"][0]
        if kind != "temporal_mixture":
            raise ValueError(f"unsupported model kind: {kind}")
        return cls(
            mode_global_probs=payload["mode_global_probs"],
            mode_position_probs=payload["mode_position_probs"],
            mode_cluster_probs=payload["mode_cluster_probs"],
            global_token_probs=payload["global_token_probs"],
            row_token_probs=payload["row_token_probs"],
            position_token_probs=payload["position_token_probs"],
            cluster_token_probs=payload["cluster_token_probs"],
            cluster_position_token_probs=payload["cluster_position_token_probs"],
            lag_transition_probs=payload["lag_transition_probs"],
            row_lag_transition_probs=payload["row_lag_transition_probs"],
            lag_steps=payload["lag_steps"],
            token_component_weights=payload["token_component_weights"].astype(np.float32, copy=False),
            mode_component_weights=payload["mode_component_weights"].astype(np.float32, copy=False),
            warmup_frames=int(payload["warmup_frames"][0]),
        )


def _context_cluster(context_frames: np.ndarray) -> int:
    if context_frames.shape[0] < 2:
        return 0
    lag1_copy_rate = float(np.mean(context_frames[-1] == context_frames[-2]))
    lag2_copy_rate = float(np.mean(context_frames[-1] == context_frames[-3])) if context_frames.shape[0] >= 3 else 0.0
    return int(lag1_copy_rate >= COPY_RATE_THRESHOLDS[0]) * 2 + int(lag2_copy_rate >= COPY_RATE_THRESHOLDS[1])


def _normalize_counts(counts: np.ndarray, smoothing: float) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    numerators = counts + smoothing
    return numerators / numerators.sum()


def _normalize_rows(counts: np.ndarray, smoothing: float) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    numerators = counts + smoothing
    denominators = numerators.sum(axis=-1, keepdims=True)
    return numerators / denominators


def _derive_token_weights(lag_steps: np.ndarray) -> np.ndarray:
    lag_strength = 1.0 / np.sqrt(lag_steps.astype(np.float64))
    lag_strength /= lag_strength.sum()
    base = np.array([0.04, 0.10, 0.06, 0.08, 0.14], dtype=np.float64)
    shared_lag = 0.34 * lag_strength
    row_lag = 0.24 * lag_strength
    weights = np.concatenate([base, shared_lag, row_lag])
    return (weights / weights.sum()).astype(np.float32)
