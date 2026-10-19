"""Join frozen per-window encoder banks to immutable HoloAssist labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file

DIMENSIONS = {"video": 768, "audio": 384, "text": 384}


def ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_feature_bank(path: Path, modality: str) -> tuple[list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_id", "features", "metadata_json"}
        if set(archive.files) != required:
            raise ValueError(f"{modality} feature bank has unexpected keys")
        ids = list(map(str, archive["sample_id"].tolist()))
        features = archive["features"]
        raw_metadata = archive["metadata_json"].item()
    metadata = json.loads(str(raw_metadata))
    if features.shape != (len(ids), DIMENSIONS[modality]) or features.dtype != np.float32:
        raise ValueError(f"invalid {modality} feature shape or dtype")
    if not ids or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError(f"{modality} feature IDs must be unique and nonempty")
    if not np.isfinite(features).all():
        raise ValueError(f"{modality} feature bank contains non-finite values")
    for field in (
        "model_id",
        "revision",
        "weights_sha256",
        "loaded_weights_sha256",
        "preprocessing",
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(f"{modality} bank lacks pinned {field}")
    if len(metadata["weights_sha256"]) != 64:
        raise ValueError(f"{modality} weights SHA-256 is invalid")
    return ids, features, metadata


def pack_split(
    rows: list[dict[str, Any]], banks: dict[str, tuple[list[str], np.ndarray, dict]], split: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ids = [str(row["id"]) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{split} windows must have unique nonempty IDs")
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"window row split differs from {split}")
    for modality, (bank_ids, _, _) in banks.items():
        if bank_ids != ids:
            raise ValueError(f"{modality} feature ID order does not match {split} windows")
    session_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session_rows.setdefault(str(row["session_id"]), []).append(row)
    for session, session_items in session_rows.items():
        ordered = sorted(session_items, key=lambda row: int(row["window_index"]))
        if [int(row["window_index"]) for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"non-contiguous temporal windows for {session}")
        if len({str(row["media_sha256"]) for row in ordered}) != 1:
            raise ValueError(f"media identity changes within {session}")
        if len({str(row["task_type"]) for row in ordered}) != 1:
            raise ValueError(f"task type changes within {session}")
    targets = np.asarray(
        [-1 if row["target"] is None else int(row["target"]) for row in rows], dtype=np.int64
    )
    if not set(np.unique(targets)).issubset({-1, 0, 1}):
        raise ValueError("binary targets expected")
    arrays = {}
    for modality, (bank_ids, bank_features, _) in banks.items():
        positions = {sample_id: index for index, sample_id in enumerate(bank_ids)}
        if not set(ids) <= positions.keys():
            raise ValueError(f"{modality} feature bank is missing {split} windows")
        arrays[modality] = np.ascontiguousarray(
            bank_features[[positions[sample_id] for sample_id in ids]], dtype=np.float32
        )
    arrays.update(
        target=targets,
        loss_mask=targets >= 0,
        sample_id=np.asarray(ids, dtype=str),
        session_id=np.asarray([str(row["session_id"]) for row in rows], dtype=str),
        window_index=np.asarray([int(row["window_index"]) for row in rows], dtype=np.int64),
        media_sha256=np.asarray([str(row["media_sha256"]) for row in rows], dtype=str),
        task_type=np.asarray([str(row["task_type"]) for row in rows], dtype=str),
    )
    report = {
        "window_count": len(rows),
        "labelled_window_count": int(np.count_nonzero(targets >= 0)),
        "unlabelled_context_window_count": int(np.count_nonzero(targets < 0)),
        "session_count": len(session_rows),
        "unique_media_count": len({str(row["media_sha256"]) for row in rows}),
        "positive_labelled_window_count": int(np.count_nonzero(targets == 1)),
        "negative_labelled_window_count": int(np.count_nonzero(targets == 0)),
        "task_type_window_counts": dict(
            sorted(Counter(str(row["task_type"]) for row in rows).items())
        ),
        "ordered_sample_ids_sha256": ordered_id_hash(ids),
    }
    return arrays, report


def select_split_banks(
    rows: list[dict[str, Any]], banks: dict[str, tuple[list[str], np.ndarray, dict]]
) -> dict[str, tuple[list[str], np.ndarray, dict]]:
    """Select split rows from the globally ordered train+validation feature banks."""
    ids = [str(row["id"]) for row in rows]
    selected = {}
    for modality, (bank_ids, features, metadata) in banks.items():
        positions = {sample_id: index for index, sample_id in enumerate(bank_ids)}
        missing = set(ids) - positions.keys()
        if missing:
            raise ValueError(f"{modality} feature bank is missing split IDs: {sorted(missing)[:3]}")
        selected[modality] = (
            ids,
            np.ascontiguousarray(features[[positions[sample_id] for sample_id in ids]]),
            metadata,
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--video-bank", type=Path, required=True)
    parser.add_argument("--audio-bank", type=Path, required=True)
    parser.add_argument("--text-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite packed features: {args.output_dir}")
    banks = {
        name: load_feature_bank(path, name)
        for name, path in (
            ("video", args.video_bank),
            ("audio", args.audio_bank),
            ("text", args.text_bank),
        )
    }
    rows = {
        split: read_jsonl(args.windows_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    expected_ids = [
        str(row["id"])
        for split in ("train", "validation")
        for row in rows[split]
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("train/validation window IDs overlap")
    for name, (bank_ids, _, _) in banks.items():
        if bank_ids != expected_ids:
            raise ValueError(f"{name} bank IDs/order do not match the frozen windows")
    packed = {}
    report = {}
    for split in ("train", "validation"):
        split_banks = select_split_banks(rows[split], banks)
        packed[split], report[split] = pack_split(rows[split], split_banks, split)
    train_sessions = set(map(str, packed["train"]["session_id"]))
    validation_sessions = set(map(str, packed["validation"]["session_id"]))
    train_assets = set(map(str, packed["train"]["media_sha256"]))
    validation_assets = set(map(str, packed["validation"]["media_sha256"]))
    if train_sessions & validation_sessions or train_assets & validation_assets:
        raise ValueError("train/validation session or media identity overlap")

    args.output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        np.savez_compressed(args.output_dir / f"{split}.npz", **packed[split])
    receipt = {
        "status": "PASS_HOLOASSIST_FROZEN_FEATURE_CACHE",
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "windows_receipts": {
            split: sha256_file(args.windows_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "input_feature_banks": {
            name: {
                "path": str(path.resolve(strict=True)),
                "sha256": sha256_file(path),
                "model": banks[name][2],
            }
            for name, path in (
                ("video", args.video_bank),
                ("audio", args.audio_bank),
                ("text", args.text_bank),
            )
        },
        "train_validation_session_overlap": 0,
        "train_validation_media_overlap": 0,
        "test_data_loaded": False,
        "report": report,
        "packed_feature_sha256": {
            split: sha256_file(args.output_dir / f"{split}.npz")
            for split in ("train", "validation")
        },
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
