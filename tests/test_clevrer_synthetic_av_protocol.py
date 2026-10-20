from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.create_clevrer_synthetic_av_protocol as protocol_module
from scripts.create_clevrer_synthetic_av_protocol import (
    eligible_collision_events,
    scene_group_split,
    scene_time_shifts,
)


def _annotation(scene: int = 17) -> dict[str, object]:
    objects = [
        {"object_id": 0, "color": "red", "material": "metal", "shape": "cube"},
        {"object_id": 1, "color": "blue", "material": "rubber", "shape": "sphere"},
        {"object_id": 2, "color": "green", "material": "rubber", "shape": "cube"},
        {"object_id": 3, "color": "purple", "material": "metal", "shape": "cylinder"},
    ]
    collisions = [
        {"object_ids": [0, 1], "frame_id": 16},
        {"object_ids": [1, 2], "frame_id": 111},
        {"object_ids": [0, 2], "frame_id": 15},
        {"object_ids": [2, 3], "frame_id": 112},
    ]
    return {"scene_index": scene, "object_property": objects, "collision": collisions}


def test_scene_split_is_deterministic_grouped_and_input_order_independent() -> None:
    ids = list(range(20))
    first = scene_group_split(ids, seed=17, validation_fraction=0.2)
    second = scene_group_split(list(reversed(ids)), seed=17, validation_fraction=0.2)

    assert first == second
    assert len(first["train"]) == 16
    assert len(first["validation"]) == 4
    assert set(first["train"]).isdisjoint(first["validation"])
    assert sorted(first["train"] + first["validation"]) == ids


def test_collision_filter_keeps_only_interior_events_and_resolves_pair_descriptors() -> None:
    events = eligible_collision_events(_annotation())

    assert [event["frame_id"] for event in events] == [16, 111]
    assert events[0]["object_descriptors"] == [
        {"color": "blue", "material": "rubber", "shape": "sphere"},
        {"color": "red", "material": "metal", "shape": "cube"},
    ]


def test_collision_filter_rejects_missing_or_ambiguous_object_pairs() -> None:
    malformed = _annotation()
    malformed["collision"] = [
        {"object_ids": [0, 99], "frame_id": 40},
        {"object_ids": [0, 1], "frame_id": 41},
    ]
    malformed["object_property"] = [
        {"object_id": 0, "color": "red", "material": "metal", "shape": "cube"},
        {"object_id": 1, "color": "blue", "material": "rubber", "shape": "sphere"},
        {"object_id": 2, "color": "blue", "material": "rubber", "shape": "sphere"},
    ]

    assert eligible_collision_events(malformed) == []


def test_collision_filter_rejects_repeated_object_pair_events() -> None:
    annotation = _annotation()
    annotation["collision"] = [
        {"object_ids": [0, 1], "frame_id": 20},
        {"object_ids": [1, 0], "frame_id": 80},
        {"object_ids": [1, 2], "frame_id": 50},
    ]

    assert [event["frame_id"] for event in eligible_collision_events(annotation)] == [50]


def test_per_scene_shifts_are_stable_independent_and_bounded() -> None:
    ids = [2, 7, 9]
    first = scene_time_shifts(ids, seed=17, max_shift_frames=255)

    assert first == scene_time_shifts(reversed(ids), seed=17, max_shift_frames=255)
    assert first == {
        scene: int(hashlib.sha256(f"17|{scene}|shift".encode()).hexdigest()[:16], 16) % 256
        for scene in ids
    }
    assert all(0 <= shift <= 255 for shift in first.values())


def test_frozen_video_encoder_preserves_sixteen_temporal_positions() -> None:
    config = json.loads(
        Path("configs/pretrained_reuse/clevrer_synthetic_av_timing_v1.json").read_text()
    )

    video = config["video_feature_extraction"]
    assert video["frame_count"] == 16
    assert video["temporal_pooling"] == "none; preserve all 16 positions"
    assert video["checkpoint_sha256"] == (
        "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
    )


def test_build_protocol_keeps_paired_variants_and_scene_groups_together(
    tmp_path, monkeypatch
) -> None:
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    entries = []
    for scene in range(10):
        path = annotation_dir / f"annotation_{scene:05d}.json"
        payload = _annotation(scene)
        payload["video_filename"] = f"video_{scene:05d}.mp4"
        payload["collision"] = [{"object_ids": [0, 1], "frame_id": 60}]
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        path.write_bytes(encoded)
        entries.append(
            {
                "scene_index": scene,
                "archive_member": f"annotations/annotation_{scene:05d}.json",
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )

    ids_sha = protocol_module.sha256_ids(range(10))
    exposure_path = tmp_path / "exposure.json"
    exposure_path.write_text(
        json.dumps({"candidate_scene_ids": list(range(10)), "candidate_scene_ids_sha256": ids_sha}),
        encoding="utf-8",
    )
    exposure_sha = hashlib.sha256(exposure_path.read_bytes()).hexdigest()
    annotation_receipt_path = tmp_path / "annotation-receipt.json"
    annotation_receipt_path.write_text(
        json.dumps(
            {
                "source_dataset": protocol_module.DATASET,
                "source_revision": protocol_module.REVISION,
                "license": "CC0-1.0",
                "split": "train",
                "validation_test_and_sealed_data_opened": False,
                "video_bytes_opened": False,
                "exposure_receipt_sha256": exposure_sha,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(protocol_module, "EXPOSURE_RECEIPT_SHA256", exposure_sha)
    monkeypatch.setattr(protocol_module, "EXPECTED_CANDIDATE_SCENE_COUNT", 10)
    monkeypatch.setattr(
        protocol_module,
        "ANNOTATION_RECEIPT_SHA256",
        hashlib.sha256(annotation_receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(protocol_module, "SCENE_IDS_SHA256", ids_sha)

    protocol = protocol_module.build_protocol(
        annotation_dir, exposure_path, annotation_receipt_path, seed=17
    )

    assert protocol["split"]["event_counts"] == {"train": 8, "validation": 2}
    assert protocol["split"]["sample_counts"] == {"train": 16, "validation": 4}
    with pytest.raises(ValueError, match="immutable seed 17"):
        protocol_module.build_protocol(
            annotation_dir, exposure_path, annotation_receipt_path, seed=18
        )
    assert set(protocol["split"]["scene_ids"]["train"]).isdisjoint(
        protocol["split"]["scene_ids"]["validation"]
    )
    for rows in protocol["samples"].values():
        grouped = {}
        for row in rows:
            grouped.setdefault(row["pair_id"], []).append(row)
        assert all(len(pair) == 2 for pair in grouped.values())
        assert all(pair[0]["question"] == pair[1]["question"] for pair in grouped.values())
        assert all({row["target"] for row in pair} == {0, 1} for pair in grouped.values())
        assert all(
            abs(pair[0]["sound_onset_frame"] - pair[1]["sound_onset_frame"]) == 32
            for pair in grouped.values()
        )
