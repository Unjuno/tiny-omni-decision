from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_egoviz_unseen_group_features import aggregate, validate_session

PLAN_FORMAT = "egoviz-unseen-session-temporal-correspondence-plan-v1"
VIDEO_CHECKPOINT = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
SENSOR_CHECKPOINT = "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
SENSOR_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _plan() -> dict[str, object]:
    return {
        "format": PLAN_FORMAT,
        "seed": 20261029,
        "splits": {
            "train": [
                {
                    "chunk_id": "train-chunk",
                    "group_id": "train-group",
                    "category": "cleaning_kitchen",
                    "expected_bytes": 100,
                    "mcap_path": "data/train/train-chunk.mcap",
                }
            ],
            "development": [
                {
                    "chunk_id": "dev-chunk",
                    "group_id": "dev-group",
                    "category": "cleaning_kitchen",
                    "expected_bytes": 110,
                    "mcap_path": "data/development/dev-chunk.mcap",
                }
            ],
            "sealed_audit": [
                {
                    "chunk_id": "sealed-chunk",
                    "group_id": "sealed-group",
                    "category": "cleaning_kitchen",
                    "expected_bytes": 120,
                    "mcap_path": "data/sealed/sealed-chunk.mcap",
                }
            ],
        },
    }


def _write_chunk(
    root: Path, plan_hash: str, *, split: str, chunk_id: str, group_id: str
) -> None:
    chunk_root = root / chunk_id
    chunk_root.mkdir(parents=True)
    plan_row = {
        "chunk_id": chunk_id,
        "group_id": group_id,
        "category": "cleaning_kitchen",
        "expected_bytes": 100 if split == "train" else 110,
        "mcap_path": f"data/{split}/{chunk_id}.mcap",
        "split": split,
        "bytes": 100 if split == "train" else 110,
        "sha256": hashlib.sha256(chunk_id.encode()).hexdigest(),
    }
    _write_json(
        chunk_root / "acquisition-manifest.json",
        {
            "format": "egoviz-unseen-group-media-acquisition-v1",
            "status": "ACQUIRED_TRAIN_DEVELOPMENT_ONLY",
            "plan_sha256": plan_hash,
            "sealed_audit_downloaded": False,
            "files": [plan_row],
        },
    )

    corpus = chunk_root / "window-corpus"
    corpus.mkdir()
    sample_id = hashlib.sha256(f"17:{chunk_id}:0".encode()).hexdigest()[:24]
    sample = {
        "sample_id": sample_id,
        "split": split,
        "group_id": group_id,
        "category": "cleaning_kitchen",
        "chunk_id": chunk_id,
        "query_index": 0,
        "query_interval_ns": [0, 2_000_000_000],
        "video_frame_indices": list(range(8)),
        "candidate_window_indices": [3, 0, 1, 2],
        "target_option_index": 1,
    }
    for target_split in ("train", "development"):
        (corpus / f"{target_split}-samples.jsonl").write_text(
            (json.dumps(sample, separators=(",", ":")) + "\n") if target_split == split else "",
            encoding="utf-8",
        )
    np.save(corpus / "imu-windows.npy", np.zeros((4, 6, 128), dtype=np.float32))
    corpus_hashes = {
        filename: _hash(corpus / filename)
        for filename in ("imu-windows.npy", "train-samples.jsonl", "development-samples.jsonl")
    }
    _write_json(
        corpus / "window-corpus.json",
        {
            "format": "egoviz-vs-sync-window-corpus-v1",
            "seed": 17,
            "window_seconds": 2,
            "stride_seconds": 1,
            "video_frames_per_window": 8,
            "imu_points_per_window": 128,
            "train_examples": int(split == "train"),
            "development_examples": int(split == "development"),
            "train_groups": int(split == "train"),
            "development_groups": int(split == "development"),
            "group_overlap": 0,
            "unique_imu_windows": 4,
            "tensor_shape": [4, 6, 128],
            "publisher_action_labels_read": False,
            "sealed_audit_accessed": False,
            "inventory": [
                {
                    "chunk_id": chunk_id,
                    "group_id": group_id,
                    "split": split,
                    "video_frames": 8,
                    "imu_samples": 128,
                    "windows": 4,
                }
            ],
            "sha256": corpus_hashes,
        },
    )

    video_dir = chunk_root / "video-features"
    video_dir.mkdir()
    video_cache = video_dir / "video-features.npz"
    np.savez_compressed(
        video_cache,
        sample_id=np.asarray([sample_id]),
        split=np.asarray([split]),
        group_id=np.asarray([group_id]),
        features=np.zeros((1, 8, 768), dtype=np.float32),
    )
    _write_json(
        video_dir / "feature-report.json",
        {
            "status": "FROZEN_FEATURES_EXTRACTED_NO_TRAINING_OR_QUALITY_EVALUATION",
            "source_checkpoint_sha256": VIDEO_CHECKPOINT,
            "upstream_commit": VIDEO_COMMIT,
            "jax_version": "0.6.2",
            "devices": ["TFRT_CPU_0"],
            "runner_sha256": _hash(
                Path(__file__).parents[1]
                / "scripts"
                / "extract_egoviz_vs_sync_video_features.py"
            ),
            "feature_file_sha256": _hash(video_cache),
            "sample_count": 1,
            "sealed_audit_accessed": False,
            "training": False,
        },
    )

    sensor_dir = chunk_root / "sensor-features"
    sensor_dir.mkdir()
    sensor_cache = sensor_dir / "sensor-features.npz"
    np.savez_compressed(
        sensor_cache,
        window_index=np.arange(4, dtype=np.int32),
        features=np.zeros((4, 6, 16, 512), dtype=np.float32),
    )
    _write_json(
        sensor_dir / "feature-report.json",
        {
            "status": "FROZEN_FEATURES_EXTRACTED_NO_TRAINING_OR_QUALITY_EVALUATION",
            "checkpoint_sha256": SENSOR_CHECKPOINT,
            "moment_source_commit": SENSOR_COMMIT,
            "device": "cpu",
            "torch_version": "2.6.0+cu124",
            "transformers_version": "5.6.2",
            "runner_sha256": _hash(
                Path(__file__).parents[1]
                / "scripts"
                / "extract_egoviz_vs_sync_sensor_features.py"
            ),
            "feature_file_sha256": _hash(sensor_cache),
            "source_window_count": 4,
            "sealed_audit_accessed": False,
            "training": False,
        },
    )


def test_aggregate_rebases_sensor_candidate_indices_and_excludes_sealed(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = _plan()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = _hash(plan_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_chunk(
        runs,
        plan_hash,
        split="train",
        chunk_id="train-chunk",
        group_id="train-group",
    )
    _write_chunk(
        runs,
        plan_hash,
        split="development",
        chunk_id="dev-chunk",
        group_id="dev-group",
    )

    output = tmp_path / "aggregate"
    report = aggregate(plan_path, runs, output)

    train = json.loads((output / "train-samples.jsonl").read_text().strip())
    dev = json.loads((output / "development-samples.jsonl").read_text().strip())
    assert train["candidate_window_indices"] == [3, 0, 1, 2]
    assert dev["candidate_window_indices"] == [7, 4, 5, 6]
    assert report["session_count"] == 2
    assert report["sealed_audit_accessed"] is False
    aggregate_corpus = json.loads((output / "window-corpus.json").read_text())
    assert aggregate_corpus["group_overlap"] == 0
    assert aggregate_corpus["train_groups"] == aggregate_corpus["development_groups"] == 1
    assert aggregate_corpus["sha256"]["imu-windows.npy"] == _hash(output / "imu-windows.npy")
    with np.load(output / "sensor-features.npz", allow_pickle=False) as cache:
        assert cache["window_index"].tolist() == list(range(8))
        assert cache["features"].shape == (8, 6, 16, 512)
    with np.load(output / "video-features.npz", allow_pickle=False) as cache:
        assert cache["sample_id"].tolist() == sorted(
            hashlib.sha256(f"17:{chunk_id}:0".encode()).hexdigest()[:24]
            for chunk_id in ("train-chunk", "dev-chunk")
        )


def test_aggregate_rejects_missing_session_and_does_not_create_output(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()

    with pytest.raises(ValueError, match="missing planned sessions"):
        aggregate(plan_path, runs, tmp_path / "aggregate")

    assert not (tmp_path / "aggregate").exists()


def test_aggregate_rejects_sealed_session_directory(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = _plan()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = _hash(plan_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_chunk(runs, plan_hash, split="train", chunk_id="train-chunk", group_id="train-group")
    _write_chunk(
        runs,
        plan_hash,
        split="development",
        chunk_id="dev-chunk",
        group_id="dev-group",
    )
    (runs / "sealed-chunk").mkdir()

    with pytest.raises(ValueError, match="unexpected session dirs"):
        aggregate(plan_path, runs, tmp_path / "aggregate")


def test_aggregate_rejects_target_not_pointing_to_matching_timestamp(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = _plan()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = _hash(plan_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_chunk(runs, plan_hash, split="train", chunk_id="train-chunk", group_id="train-group")
    _write_chunk(
        runs,
        plan_hash,
        split="development",
        chunk_id="dev-chunk",
        group_id="dev-group",
    )
    sample_path = runs / "train-chunk" / "window-corpus" / "train-samples.jsonl"
    sample = json.loads(sample_path.read_text().strip())
    sample["candidate_window_indices"][sample["target_option_index"]] = 2
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    corpus_path = sample_path.parent / "window-corpus.json"
    corpus = json.loads(corpus_path.read_text())
    corpus["sha256"]["train-samples.jsonl"] = _hash(sample_path)
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    with pytest.raises(ValueError, match="target/window references"):
        aggregate(plan_path, runs, tmp_path / "aggregate")

    assert not (tmp_path / "aggregate").exists()


def test_validate_session_checks_only_selected_planned_session(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan = _plan()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = _hash(plan_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_chunk(runs, plan_hash, split="train", chunk_id="train-chunk", group_id="train-group")
    (runs / "sealed-chunk").mkdir()

    report = validate_session(plan_path, runs, "train-chunk")

    assert report["status"] == "FROZEN_SESSION_PACKAGE_VALIDATED_NO_TRAINING_OR_QUALITY_EVALUATION"
    assert report["split"] == "train"
    assert report["sample_count"] == 1
    assert report["sealed_audit_accessed"] is False
    assert not any((runs / "sealed-chunk").iterdir())


def test_validate_session_rejects_sealed_or_unknown_chunk_before_opening_it(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    sealed = runs / "sealed-chunk"
    sealed.mkdir()
    marker = sealed / "must-not-read.json"
    marker.write_text("not opened", encoding="utf-8")

    with pytest.raises(ValueError, match="not uniquely present"):
        validate_session(plan_path, runs, "sealed-chunk")

    assert marker.read_text(encoding="utf-8") == "not opened"
