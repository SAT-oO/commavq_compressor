from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np


def _progress(iterable, *, total: int | None, desc: str | None, unit: str):
    if desc is None:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


def default_eval_shards(repo_root: Path) -> list[Path]:
    dataset_root = repo_root / "resource" / "dataset"
    return [dataset_root / "data-0000.tar.gz", dataset_root / "data-0001.tar.gz"]


def iter_shard_records(
    shard_paths: list[Path],
    limit: int | None = None,
    per_shard_limit: int | None = None,
    progress_desc: str | None = None,
    progress_total: int | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    emitted = 0
    shard_iter = _progress(shard_paths, total=len(shard_paths), desc=None, unit="shard")
    progress = None
    if progress_desc is not None:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=progress_total, desc=progress_desc, unit="record")
        except Exception:
            progress = None

    for shard_path in shard_iter:
        emitted_in_shard = 0
        with tarfile.open(shard_path, "r:gz") as archive:
            grouped: dict[str, dict[str, tarfile.TarInfo]] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                if member.name.endswith(".token.npy"):
                    grouped.setdefault(member.name[:-10], {})["token"] = member
                elif member.name.endswith(".json"):
                    grouped.setdefault(member.name[:-5], {})["json"] = member

            for base_name in sorted(grouped):
                members = grouped[base_name]
                if "token" not in members or "json" not in members:
                    continue

                token_blob = archive.extractfile(members["token"]).read()
                meta_blob = archive.extractfile(members["json"]).read()
                tokens = np.load(io.BytesIO(token_blob)).astype(np.uint16, copy=False)
                file_name = json.loads(meta_blob)["file_name"]
                yield file_name, tokens

                emitted += 1
                emitted_in_shard += 1
                if progress is not None:
                    progress.update(1)
                if limit is not None and emitted >= limit:
                    if progress is not None:
                        progress.close()
                    return
                if per_shard_limit is not None and emitted_in_shard >= per_shard_limit:
                    break

    if progress is not None:
        progress.close()


def count_shard_records(
    shard_paths: list[Path],
    limit: int | None = None,
    per_shard_limit: int | None = None,
) -> int:
    emitted = 0
    for shard_path in shard_paths:
        emitted_in_shard = 0
        with tarfile.open(shard_path, "r:gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".token.npy"):
                    emitted += 1
                    emitted_in_shard += 1
                    if limit is not None and emitted >= limit:
                        return emitted
                    if per_shard_limit is not None and emitted_in_shard >= per_shard_limit:
                        break
    return emitted
