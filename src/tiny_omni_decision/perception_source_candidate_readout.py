"""Small frozen-feature cross-modal source-candidate readout utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateFeatureExample:
    record_id: str
    video_id: str
    candidate_ids: tuple[str, ...]
    visual_mean_delta: np.ndarray
    audio: np.ndarray
    offset_audio: np.ndarray
    donor_record_id: str
    target_candidate_ids: frozenset[str]
    none_visible: bool


def pool_audio_features(references, features, *, variant: str) -> dict[str, np.ndarray]:
    """Mean-pool unique windows for each record without crossing split rows."""
    grouped: dict[str, list[np.ndarray]] = {}
    seen: set[tuple[str, int]] = set()
    values = np.asarray(features)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("audio features must be a finite 2D array")
    for row in references:
        if row.get("variant") != variant:
            continue
        record_id = row["record_id"]
        feature_index = int(row["feature_index"])
        key = (record_id, feature_index)
        if key in seen:
            continue
        if not 0 <= feature_index < len(values):
            raise ValueError("audio feature index is out of range")
        seen.add(key)
        grouped.setdefault(record_id, []).append(values[feature_index])
    pooled = {}
    for record_id, rows in grouped.items():
        value = np.mean(np.stack(rows), axis=0, dtype=np.float64).astype(np.float32)
        value /= max(float(np.linalg.norm(value)), 1e-12)
        pooled[record_id] = value
    return pooled


def pool_video_candidate_features(records, feature_index_rows, features):
    """Pool ordered ROI features as [temporal mean, last-minus-first]."""
    values = np.asarray(features)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("video features must be a finite 2D array")
    lookup = {}
    for row in feature_index_rows:
        key = (row["record_id"], row["candidate_id"], int(row["frame_id"]))
        if key in lookup:
            raise ValueError(f"duplicate video feature index: {key}")
        index = int(row["feature_index"])
        if not 0 <= index < len(values):
            raise ValueError("video feature index is out of range")
        lookup[key] = values[index]

    pooled = {}
    for record in records:
        record_id = record["record_id"]
        candidates = {}
        for candidate in record["candidates"]:
            candidate_id = candidate["candidate_id"]
            ordered = sorted(candidate["frames"], key=lambda frame: int(frame["frame_id"]))
            frame_values = []
            for frame in ordered:
                key = (record_id, candidate_id, int(frame["frame_id"]))
                if key not in lookup:
                    raise ValueError(f"video ROI feature missing: {key}")
                frame_values.append(lookup[key])
            if not frame_values:
                raise ValueError(f"candidate has no annotated frame: {record_id}/{candidate_id}")
            sequence = np.stack(frame_values).astype(np.float32, copy=False)
            mean = sequence.mean(axis=0, dtype=np.float64).astype(np.float32)
            delta = (sequence[-1] - sequence[0]).astype(np.float32)
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            delta /= max(float(np.linalg.norm(delta)), 1e-12)
            candidates[candidate_id] = np.concatenate((mean, delta))
        pooled[record_id] = candidates
    return pooled


def build_feature_examples(
    plan_records,
    labels_by_id,
    video_candidate_features,
    source_audio_by_id,
    offset_audio_by_id,
):
    """Join frozen label-free features with only the requested split labels."""
    result = []
    plan_ids = set()
    for row in plan_records:
        record_id = row["record_id"]
        if record_id in plan_ids:
            raise ValueError(f"duplicate feature-plan record: {record_id}")
        plan_ids.add(record_id)
        label = labels_by_id.get(record_id)
        if label is None:
            raise ValueError(f"split label missing for feature-plan record: {record_id}")
        candidate_ids = tuple(candidate["candidate_id"] for candidate in row["candidates"])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate ID: {record_id}")
        visual = video_candidate_features.get(record_id)
        if visual is None or set(visual) != set(candidate_ids):
            raise ValueError(f"visual candidates differ from plan: {record_id}")
        if record_id not in source_audio_by_id or record_id not in offset_audio_by_id:
            raise ValueError(f"audio source/offset features missing: {record_id}")
        if any(np.asarray(visual[key]).shape != (1536,) for key in candidate_ids):
            raise ValueError(
                f"visual feature dimension differs from VideoPrism ROI contract: {record_id}"
            )
        if (
            np.asarray(source_audio_by_id[record_id]).shape != (384,)
            or np.asarray(offset_audio_by_id[record_id]).shape != (384,)
        ):
            raise ValueError(
                f"audio feature dimension differs from EfficientAT contract: {record_id}"
            )
        targets = frozenset(label["supervision"]["target_candidate_ids"])
        none_visible = bool(label["supervision"]["none_visible"])
        if none_visible and targets:
            raise ValueError(f"none-visible record has positive candidates: {record_id}")
        if not none_visible and (not targets or not targets.issubset(candidate_ids)):
            raise ValueError(f"target candidates do not match plan: {record_id}")
        result.append(
            CandidateFeatureExample(
                record_id=record_id,
                video_id=row["video_id"],
                candidate_ids=candidate_ids,
                visual_mean_delta=np.stack([visual[key] for key in candidate_ids]),
                audio=np.asarray(source_audio_by_id[record_id], dtype=np.float32),
                offset_audio=np.asarray(offset_audio_by_id[record_id], dtype=np.float32),
                donor_record_id=row["audio_shuffle_donor_record_id"],
                target_candidate_ids=targets,
                none_visible=none_visible,
            )
        )
        if row.get("audio_shuffle_donor_video_id") == row["video_id"]:
            raise ValueError(f"audio shuffle donor is from the same video: {record_id}")
    if plan_ids != set(labels_by_id):
        raise ValueError("feature-plan IDs and fixed split label IDs differ")
    if plan_ids != set(source_audio_by_id) or plan_ids != set(offset_audio_by_id):
        raise ValueError("audio feature IDs and fixed split IDs differ")
    return result
