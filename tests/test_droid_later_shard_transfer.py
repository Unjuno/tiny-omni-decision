from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import build_droid_later_shard_holdout_task as builder
from scripts.evaluate_droid_frozen_readout_transfer import (
    evaluate_saved_readouts,
    validate_feature_progress,
)
from tiny_omni_decision.droid_next_action_readout import ACTION_OPTIONS


def test_build_manifest_keeps_frozen_later_shards_evaluation_only(
    tmp_path: Path, monkeypatch
) -> None:
    indices = [466, 759, 1632]
    shard_paths = {}
    shard_hashes = {}
    records_by_path = {}
    audit_shards = []
    objects = []
    for index in indices:
        filename = f"shard-{index:05d}-of-02048.tfrecord"
        path = tmp_path / filename
        raw = f"shard {index}".encode()
        path.write_bytes(raw)
        shard_paths[index] = path
        shard_hashes[index] = hashlib.sha256(raw).hexdigest()
        episode = f"episode-{index}"
        records_by_path[path.name] = [
            {
                "episode": episode,
                "context": {
                    "episode_metadata/file_path": f"x/r2d2-data-full/site{index}/{episode}",
                    "episode_metadata/recording_folderpath": (
                        f"x/r2d2-data-full/site{index}/recording{index}"
                    ),
                },
            }
        ]
        audit_shards.append(
            {"index": index, "records": 1, "eligible_records": 1,
             "bytes": len(raw), "sha256": shard_hashes[index]}
        )
        objects.append(
            {"index": index,
             "object": f"robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-{index:05d}-of-02048",
             "size": len(raw)}
        )

    def fake_context(record):
        return record["context"]

    def fake_float_values(record, feature_name):
        if feature_name == "steps/action_dict/cartesian_velocity":
            values = np.zeros((32, 6), dtype=np.float32)
            values[8, 0] = 1.0
            return values.reshape(-1).tolist()
        width = {
            "steps/observation/joint_position": 7,
            "steps/observation/cartesian_position": 6,
            "steps/observation/gripper_position": 1,
        }[feature_name]
        return np.zeros(32 * width, dtype=np.float32).tolist()

    def fake_bytes_values(record, feature_name):
        return [f"{record['episode']}:{feature_name}:{step}".encode() for step in range(32)]

    monkeypatch.setattr(builder, "episode_example_context", fake_context)
    monkeypatch.setattr(builder, "episode_example_float_values", fake_float_values)
    monkeypatch.setattr(builder, "episode_example_bytes_values", fake_bytes_values)
    monkeypatch.setattr(
        builder,
        "iter_tfrecord",
        lambda path: iter(records_by_path[Path(path).name]),
    )

    audit_path = tmp_path / "source-audit.json"
    audit_path.write_text(
        json.dumps({
            "status": "PASS_SOURCE_METADATA_AND_IDENTITY",
            "labels_read": False,
            "media_decoded": False,
            "sealed_audit_loaded": False,
            "selection_indices": indices,
            "selection_sha256": "selection",
            "new_sites_in_selected_sample": [],
            "eligible_count_in_selected_shards": len(indices),
            "shards": audit_shards,
        }),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"indices": indices, "objects": objects}), encoding="utf-8"
    )
    previous_path = tmp_path / "previous.json"
    previous_path.write_text(
        json.dumps({
            "splits": {
                "train": {"rows": [{"id": "old-id", "group_id": "old-group"}],
                          "site_counts": {"prior": 1}},
                "validation": {"rows": [], "site_counts": {}},
            }
        }),
        encoding="utf-8",
    )

    output = tmp_path / "frozen-task.json"
    task = builder.build_manifest(
        inputs=[(index, shard_paths[index]) for index in indices],
        source_audit_path=audit_path,
        selection_path=selection_path,
        previous_task_path=previous_path,
        output_path=output,
    )

    rows = task["splits"]["validation"]["rows"]
    assert len(rows) == 3
    assert task["splits"]["train"]["rows"] == []
    assert task["model_training"] is False
    assert task["sealed_audit_loaded"] is False
    assert task["split_protocol"]["source_group_overlap_with_previous_task"] == 0
    assert all(row["target_index"] == 8 for row in rows)
    assert all(row["target_option_id"] == "x+" for row in rows)
    assert all(row["split"] == "validation" for row in rows)
    assert len({row["group_id"] for row in rows}) == 3


def test_saved_readout_transfer_evaluation_reports_site_and_shard_metrics() -> None:
    count = 4
    task = {
        "model_training": False,
        "sealed_audit_loaded": False,
        "splits": {
            "validation": {
                "rows": [
                    {"id": f"id-{i}", "group_id": f"group-{i}",
                     "split": "validation", "site": "site-a" if i < 2 else "site-b",
                     "shard_index": 466 if i < 2 else 759,
                     "target_option_id": ACTION_OPTIONS[i % 2]}
                    for i in range(count)
                ]
            }
        },
    }
    readouts = {}
    for arm, width in (("video", 1536), ("sensor", 112), ("video_sensor", 1648)):
        bias = np.zeros(len(ACTION_OPTIONS), dtype=np.float64)
        bias[0] = 1.0
        readouts[arm] = {
            "mean": np.zeros(width),
            "scale": np.ones(width),
            "weight": np.zeros((len(ACTION_OPTIONS), width)),
            "bias": bias,
        }

    result = evaluate_saved_readouts(
        task=task,
        video=np.zeros((count, 3, 8, 768), dtype=np.float32),
        sensor=np.zeros((count, 8, 14), dtype=np.float32),
        readouts=readouts,
    )

    assert result["evaluation_count"] == 4
    assert result["unique_sites"] == 2
    assert result["unique_recording_groups"] == 4
    assert result["metrics"]["video"]["per_site"]["site-a"]["count"] == 2
    assert result["metrics"]["sensor"]["per_shard"]["759"]["count"] == 2
    assert len(result["predictions"]) == 4


def test_saved_readout_evaluation_rejects_group_duplicates() -> None:
    task = {
        "model_training": False,
        "sealed_audit_loaded": False,
        "splits": {"validation": {"rows": [
            {"id": "a", "group_id": "same", "split": "validation", "site": "s",
             "shard_index": 466, "target_option_id": "stop"},
            {"id": "b", "group_id": "same", "split": "validation", "site": "s",
             "shard_index": 466, "target_option_id": "stop"},
        ]}},
    }
    readout = {
        "mean": np.zeros(1536), "scale": np.ones(1536),
        "weight": np.zeros((7, 1536)), "bias": np.zeros(7),
    }
    readouts = {
        "video": readout,
        "sensor": {
            **readout,
            "mean": np.zeros(112),
            "scale": np.ones(112),
            "weight": np.zeros((7, 112)),
        },
        "video_sensor": {
            **readout,
            "mean": np.zeros(1648),
            "scale": np.ones(1648),
            "weight": np.zeros((7, 1648)),
        },
    }
    with np.testing.assert_raises_regex(ValueError, "recording groups must be unique"):
        evaluate_saved_readouts(
            task=task,
            video=np.zeros((2, 3, 8, 768)),
            sensor=np.zeros((2, 8, 14)),
            readouts=readouts,
        )


def test_progress_is_validated_by_row_index_not_append_order() -> None:
    rows = [
        {"id": "a", "target_option_id": "stop"},
        {"id": "b", "target_option_id": "x-"},
    ]
    progress = [
        {"index": 1, "id": "b", "split": "validation", "label": 1},
        {"index": 0, "id": "a", "split": "validation", "label": 0},
    ]

    validate_feature_progress(progress, rows)

    with np.testing.assert_raises_regex(ValueError, "duplicate row index"):
        validate_feature_progress([progress[0], progress[0]], rows)
