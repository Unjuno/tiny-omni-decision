"""TRAIN-only grouped development splits and fit-fold-only feature transforms."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np


def grouped_folds(
    train_rows: list[dict[str, Any]], *, folds: int = 5, seed: int = 17
) -> dict[str, int]:
    """Assign intact 2x2 source pairs, stratified by condition, to balanced folds.

    Only supplied TRAIN rows are inspected. Labels are used to verify the fixed
    match design, never to choose a fold. Sorting and hashing make row order
    irrelevant. Each source group must belong to exactly one source pair.
    """
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    owners: dict[str, str] = {}
    seen_samples: set[tuple[str, str]] = set()
    explicit_ids: set[str] = set()
    for row in train_rows:
        for key in ("pair_unit_id", "condition", "audio_group", "video_group"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"TRAIN row requires a nonempty {key}")
        if type(row.get("target")) is not int or row["target"] not in (0, 1):
            raise ValueError("TRAIN target must be binary")
        audio, video, pair = row["audio_group"], row["video_group"], row["pair_unit_id"]
        if (audio == video) != (row["target"] == 1):
            raise ValueError("TRAIN target disagrees with source identity")
        identity = (audio, video)
        if identity in seen_samples:
            raise ValueError("duplicate TRAIN sample identity")
        seen_samples.add(identity)
        if "sample_id" in row:
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in explicit_ids:
                raise ValueError("duplicate or invalid TRAIN sample_id")
            explicit_ids.add(sample_id)
        for group in (audio, video):
            if group in owners and owners[group] != pair:
                raise ValueError("source group belongs to multiple pair units")
            owners[group] = pair
        pairs[pair].append(row)
    if len(pairs) < folds:
        raise ValueError("at least one complete pair unit per fold is required")
    by_condition: dict[str, list[str]] = defaultdict(list)
    for pair, rows in pairs.items():
        groups = {row[key] for row in rows for key in ("audio_group", "video_group")}
        observed = {(row["audio_group"], row["video_group"]) for row in rows}
        conditions = {row["condition"] for row in rows}
        if (
            len(rows) != 4
            or len(groups) != 2
            or len(conditions) != 1
            or sum(row["target"] for row in rows) != 2
            or observed != {(audio, video) for audio in groups for video in groups}
        ):
            raise ValueError("pair unit must be one complete balanced 2x2 condition")
        by_condition[next(iter(conditions))].append(pair)

    def priority(pair: str) -> str:
        payload = json.dumps(["robomnist-train-cv", seed, pair], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    assigned: dict[str, int] = {}
    offset = 0
    for condition in sorted(by_condition):
        ordered = sorted(by_condition[condition], key=lambda pair: (priority(pair), pair))
        for index, pair in enumerate(ordered):
            assigned[pair] = (offset + index) % folds
        offset = (offset + len(ordered)) % folds
    return assigned


def pooled_normalize(
    features: dict[str, tuple[np.ndarray, np.ndarray]], fit_groups: set[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Pool exactly as the frozen baseline, fitting standardization inside a fold.

    A source contributes once to the mean/scale even if its feature is reused by
    several positive and negative rows. Held-out groups are transformed only.
    """
    if not fit_groups or not fit_groups <= features.keys():
        raise ValueError("fit_groups must be a nonempty subset of supplied feature groups")
    pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    widths = None
    for group, streams in features.items():
        if not isinstance(streams, (tuple, list)) or len(streams) != 2:
            raise ValueError("features must contain one audio and one video stream")
        audio, video = (np.asarray(stream, dtype=np.float32) for stream in streams)
        if any(
            stream.ndim != 2
            or stream.shape[0] != 8
            or stream.shape[1] < 1
            or not np.isfinite(stream).all()
            for stream in (audio, video)
        ):
            raise ValueError("features must be finite eight-step audio/video sequences")
        current_widths = (audio.shape[1], video.shape[1])
        if widths is not None and current_widths != widths:
            raise ValueError("feature widths differ between source groups")
        widths = current_widths
        pooled[group] = (
            np.concatenate((audio.mean(0), audio.std(0))),
            np.concatenate((video.mean(0), video.std(0))),
        )
    normalizers = {}
    for index, modality in enumerate(("audio", "video")):
        values = np.stack([pooled[group][index] for group in sorted(fit_groups)])
        mean, scale = values.mean(0), values.std(0)
        scale[scale < 1e-6] = 1.0
        normalizers[f"{modality}_mean"] = mean
        normalizers[f"{modality}_scale"] = scale
    transformed = {
        group: tuple(
            ((streams[index] - normalizers[f"{modality}_mean"])
             / normalizers[f"{modality}_scale"]).astype(np.float32)
            for index, modality in enumerate(("audio", "video"))
        )
        for group, streams in pooled.items()
    }
    return transformed, normalizers
