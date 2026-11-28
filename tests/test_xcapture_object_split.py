from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_xcapture_sample import choose_sample_points
from scripts.build_xcapture_object_split import (
    build_group_split,
    build_source_split,
)


def test_group_split_is_deterministic_complete_and_disjoint():
    ids = [f"{value:05d}" for value in range(12)]
    first = build_group_split(ids, seed=17, sizes={"train": 6, "validation": 3, "evaluation": 3})
    second = build_group_split(ids, seed=17, sizes={"train": 6, "validation": 3, "evaluation": 3})

    assert first == second
    assert first["partition_counts"] == {"train": 6, "validation": 3, "evaluation": 3}
    partitions = [set(values) for values in first["partitions"].values()]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert set.union(*partitions) == set(ids)


def test_group_split_fails_closed_for_duplicate_or_incomplete_assignment():
    with pytest.raises(ValueError, match="unique"):
        build_group_split(["00001", "00001"], seed=17, sizes={"train": 1, "evaluation": 1})
    with pytest.raises(ValueError, match="sum"):
        build_group_split(["00001", "00002"], seed=17, sizes={"train": 1})


def test_pinned_source_split_has_expected_group_counts_and_hashes():
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (root / "manifests/pretrained_reuse/xcapture-source-screen-20261128.json").read_text()
    )
    split = build_source_split(source)
    saved_split = json.loads(
        (root / "manifests/pretrained_reuse/xcapture-object-split-seed17.json").read_text()
    )

    assert saved_split == split
    assert split["partition_counts"] == {"train": 400, "validation": 100, "evaluation": 100}
    ids = split["partitions"]
    assert set(ids["train"]).isdisjoint(ids["validation"])
    assert set(ids["train"]).isdisjoint(ids["evaluation"])
    assert set(ids["validation"]).isdisjoint(ids["evaluation"])
    assert split["source_object_inventory"]["ids_sha256"] == (
        "eb85eafde15a12cf64eb7202d61e1504b62f201d769d08e12c445f101e514785"
    )


def test_source_split_refuses_uncleared_license_gate():
    with pytest.raises(ValueError, match="license/task gate"):
        build_source_split({"status": "REVIEW_NOT_ADMITTED"})


def test_media_sample_selection_uses_complete_train_and_validation_points_only():
    entries = {}
    for object_id in ("00001", "00002", "00003"):
        for modality, filename in (
            ("image", "rgb.png"),
            ("audio", "audio_normalized.wav"),
            ("tactile", "10N_tactile.png"),
        ):
            folder = {"image": "vision", "audio": "audio", "tactile": "tactile"}[modality]
            entries[
                f"XCapture_data/{object_id}/000/{folder}/{filename}"
            ] = {}
    split = {
        "partitions": {
            "train": ["00001"],
            "validation": ["00002"],
            "evaluation": ["00003"],
        }
    }

    points = choose_sample_points(entries, split, per_partition=1)

    assert [(item["partition"], item["object_id"]) for item in points] == [
        ("train", "00001"),
        ("validation", "00002"),
    ]
