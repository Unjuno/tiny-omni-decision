from __future__ import annotations

import pytest

from scripts.prepare_xcapture_cross_point_corpus import (
    REQUIRED_PATHS,
    _verify_plan,
    build_plan,
    choose_cross_points,
)


def _split():
    return {
        "archive_sha256": "71855c2d9cdfb6d54ed379ff4059b6c370e6f7ef040830b0ac19063159c477c2",
        "partitions": {
            "train": ["00001", "00002"],
            "validation": ["00003"],
            "evaluation": ["00004"],
        },
    }


def _entries(*, incomplete_object: str | None = None):
    entries = {}
    for object_id in ("00001", "00002", "00003", "00004"):
        for point_id in ("000", "001", "002"):
            for modality, suffix in REQUIRED_PATHS.items():
                if (
                    incomplete_object == object_id
                    and point_id != "000"
                    and modality == "audio"
                ):
                    continue
                path = f"XCapture_data/{object_id}/{point_id}/{suffix}"
                entries[path] = {
                    "method": 8,
                    "crc32": 1,
                    "compressed_size": 2,
                    "uncompressed_size": 3,
                    "local_header_offset": len(entries) * 10,
                }
    return entries


def test_choose_cross_points_is_deterministic_and_object_disjoint():
    first = choose_cross_points(_entries(), _split())
    second = choose_cross_points(_entries(), _split())

    assert first == second
    assert len(first) == 6
    assert {row["object_id"] for row in first if row["partition"] == "train"} == {
        "00001",
        "00002",
    }
    assert {row["object_id"] for row in first if row["partition"] == "validation"} == {
        "00003"
    }
    assert all(row["object_id"] != "00004" for row in first)
    assert all(
        len({row["point_id"] for row in first if row["object_id"] == object_id}) == 2
        for object_id in ("00001", "00002", "00003")
    )


def test_incomplete_object_fails_instead_of_imputing_a_point():
    with pytest.raises(ValueError, match="only 1 complete points"):
        choose_cross_points(_entries(incomplete_object="00001"), _split())


def test_plan_freezes_only_train_and_validation_assets():
    plan = build_plan(
        _entries(),
        _split(),
        split_sha256="split-hash",
        source_sha256="source-hash",
    )

    assert plan["partition_point_counts"] == {"train": 4, "validation": 2}
    assert plan["file_count"] == 24
    assert plan["evaluation_media_accessed"] is False
    assert all(row["partition"] in {"train", "validation"} for row in plan["files"])
    _verify_plan(plan)


def test_plan_hash_detects_changes():
    plan = build_plan(
        _entries(),
        _split(),
        split_sha256="split-hash",
        source_sha256="source-hash",
    )
    plan["files"].pop()

    with pytest.raises(ValueError, match="plan hash mismatch"):
        _verify_plan(plan)
