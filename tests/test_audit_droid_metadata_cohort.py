from __future__ import annotations

import base64
import hashlib
import json

import pytest

from scripts.audit_droid_metadata_cohort import (
    digest_identity,
    metadata_objects,
    normalize_object_ref,
    parse_group_metadata,
    select_sample,
    verify_payload,
    video_reference_counts,
)


def _row(name: str, generation: str = "1") -> dict[str, str]:
    payload = name.encode("utf-8")
    return {
        "name": name,
        "generation": generation,
        "size": str(len(payload)),
        "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
    }


def _json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def test_metadata_objects_filters_release_prefix_and_rejects_duplicate_names() -> None:
    rows = [
        _row("robotics/droid_raw/1.0.1/site/metadata_a.json"),
        _row("robotics/droid_raw/1.0.1/site/video.mp4"),
        _row("robotics/droid_raw/1.0.0/site/metadata_old.json"),
    ]
    assert [row["name"] for row in metadata_objects(rows, "robotics/droid_raw/1.0.1/")] == [
        "robotics/droid_raw/1.0.1/site/metadata_a.json"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        metadata_objects([rows[0], rows[0]], "robotics/droid_raw/1.0.1/")


def test_sample_order_is_deterministic_and_independent_of_input_order() -> None:
    rows = [_row(f"robotics/droid_raw/1.0.1/{index}/metadata_{index}.json") for index in range(12)]
    left = select_sample(rows, 5, seed=17)
    right = select_sample(list(reversed(rows)), 5, seed=17)
    assert [row["name"] for row in left] == [row["name"] for row in right]
    assert [row["name"] for row in left] != [row["name"] for row in select_sample(rows, 5, seed=18)]


def test_scene_group_identity_uses_building_and_scene_and_hashes_collector() -> None:
    inventory = {
        "robotics/droid_raw/1.0.1/lab-a/a.h5",
        "robotics/droid_raw/1.0.1/lab-a/cam.mp4",
    }
    first = parse_group_metadata(
        _json_bytes(
            {
                "building": "Building A",
                "lab": "lab-a",
                "scene_id": 3,
                "user_id": "person-1",
                "hdf5_path": "a.h5",
                "cam_mp4_path": "cam.mp4",
                "success": True,
                "current_task": "ignored",
            }
        ),
        inventory,
    )
    repeated = parse_group_metadata(
        _json_bytes(
            {
                "building": "building a",
                "lab": "lab-a",
                "scene_id": 3,
                "user_id": "person-2",
                "hdf5_path": "a.h5",
                "cam_mp4_path": "cam.mp4",
            }
        ),
        inventory,
    )
    other_scene = parse_group_metadata(
        _json_bytes({"building": "Building A", "scene_id": 4, "user_id": "person-1"}),
        inventory,
    )
    assert first["building_scene_id_tuple_sha256"] == repeated["building_scene_id_tuple_sha256"]
    assert first["building_scene_id_tuple_sha256"] != other_scene["building_scene_id_tuple_sha256"]
    assert first["collector_sha256"] != repeated["collector_sha256"]
    assert first["video_refs_present"] == 1
    assert first["trajectory_ref_present"] is True
    assert "current_task" not in first
    assert "success" not in first


def test_group_metadata_fails_closed_when_scene_or_collector_is_missing() -> None:
    with pytest.raises(ValueError, match="scene"):
        parse_group_metadata(_json_bytes({"building": "A", "user": "u"}), set())
    with pytest.raises(ValueError, match="collector"):
        parse_group_metadata(_json_bytes({"building": "A", "scene_id": 1}), set())


def test_object_references_are_normalized_without_fetching_payloads() -> None:
    assert normalize_object_ref("gs://gresearch/robotics/droid_raw/1.0.1/a.mp4") == (
        "robotics/droid_raw/1.0.1/a.mp4"
    )
    assert normalize_object_ref("a.mp4") == "robotics/droid_raw/1.0.1/a.mp4"
    assert normalize_object_ref("success/a.mp4", "lab-a") == (
        "robotics/droid_raw/1.0.1/lab-a/success/a.mp4"
    )
    assert normalize_object_ref(None) is None


def test_payload_verification_checks_exact_size_and_gcs_md5() -> None:
    payload = b"metadata only"
    row = {
        "size": str(len(payload)),
        "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
    }
    assert verify_payload(row, payload) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="size mismatch"):
        verify_payload(row, payload + b"x")
    bad_md5 = {**row, "md5Hash": base64.b64encode(b"0" * 16).decode("ascii")}
    with pytest.raises(ValueError, match="MD5"):
        verify_payload(bad_md5, payload)


def test_identity_hash_is_stable_and_namespaced() -> None:
    assert digest_identity(b"scene\0", "X") == digest_identity(b"scene\0", "x")
    assert digest_identity(b"scene\0", "x") != digest_identity(b"collector\0", "x")


def test_video_reference_counts_separate_aliases_from_cross_record_reuse() -> None:
    counts = video_reference_counts(
        [
            {"video_asset_sha256": ["a", "b", "b"]},
            {"video_asset_sha256": ["b", "c"]},
        ]
    )
    assert counts == {
        "reference_occurrences": 5,
        "unique_asset_references": 3,
        "within_record_alias_occurrences": 1,
        "cross_record_duplicate_occurrences": 1,
    }
