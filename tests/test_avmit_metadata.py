from __future__ import annotations

import csv
import hashlib
import json

import pytest

from tiny_omni_decision.avmit_metadata import (
    audit_avmit_metadata,
    build_savc_pair_rows,
    canonical_jsonl_bytes,
    deterministic_split,
)


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rating(filename, location="training", label="walking"):
    return {
        "filename": filename,
        "tfrecord_filename": f"records/{filename.replace('/', '_')}.tfrecord",
        "video_location": location,
        "AVMIT_label": label,
        "r1": "0",
        "r2": "0",
        "r3": "3",
    }


def test_split_is_deterministic_and_assigns_every_id():
    ids = [f"class/video-{index}.mp4" for index in range(100)]
    assert [deterministic_split(item, 17) for item in ids] == [
        deterministic_split(item, 17) for item in ids
    ]
    assert {deterministic_split(item, 17) for item in ids} == {
        "train",
        "calibration",
        "validation",
        "sealed_audit",
    }


def test_audit_excludes_published_test_and_source_validation_and_seals_targets(tmp_path):
    fields = ["filename", "tfrecord_filename", "video_location", "AVMIT_label", "r1", "r2", "r3"]
    ratings = tmp_path / "ratings.csv"
    test_set = tmp_path / "test.csv"
    output = tmp_path / "out"
    _write_csv(
        ratings,
        fields,
        [
            *[
                _rating(f"class/video-{index}.mp4", label=f"action-{index % 3}")
                for index in range(100)
            ],
            _rating(
                "heldout/video.mp4", location="validation", label="test-target-must-not-appear"
            ),
            _rating("test/video.mp4", label="test-target-must-not-appear"),
        ],
    )
    _write_csv(
        test_set,
        ["filename", "AVMIT_label", "tfrecord_filename"],
        [
            {
                "filename": "test/video.mp4",
                "AVMIT_label": "must-not-be-read",
                "tfrecord_filename": "records/test_video.mp4.tfrecord",
            }
        ],
    )

    summary = audit_avmit_metadata(ratings, test_set, output)

    assert summary["ratings_rows_by_publisher_location"] == {"training": 101, "validation": 1}
    assert summary["training_rows_excluded_by_published_test_identity"] == 1
    assert summary["eligible_source_videos"] == 100
    assert summary["cross_split_source_identity_overlap"] == 0
    assert summary["cross_split_feature_record_overlap"] == 0
    for split, values in summary["splits"].items():
        path = output / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == values["manifest_sha256"]
    sealed = (output / "sealed_audit.jsonl").read_text(encoding="utf-8")
    assert "target_label" not in sealed
    assert "test-target-must-not-appear" not in "".join(
        path.read_text(encoding="utf-8") for path in output.glob("*.json*")
    )
    assert (
        json.loads((output / "metadata-audit.json").read_text(encoding="utf-8"))["splits"][
            "sealed_audit"
        ]["rater_vote_target_counts"]
        is None
    )


def test_audit_rejects_duplicate_source_identity(tmp_path):
    fields = ["filename", "tfrecord_filename", "video_location", "AVMIT_label", "r1", "r2", "r3"]
    ratings = tmp_path / "ratings.csv"
    test_set = tmp_path / "test.csv"
    _write_csv(ratings, fields, [_rating("class/repeat.mp4"), _rating("CLASS/repeat.mp4")])
    _write_csv(test_set, ["filename"], [])

    with pytest.raises(ValueError, match="duplicate source video identity"):
        audit_avmit_metadata(ratings, test_set, tmp_path / "out")


def test_savc_pairs_are_balanced_and_each_source_is_grouped_once():
    rows = [
        {
            "asset_id_sha256": f"asset-{index}",
            "feature_record": f"features/{index}.tfrecord",
            "target_label": label,
        }
        for index, label in enumerate(("a", "a", "b", "b", "c", "c"))
    ]

    pairs, leftovers = build_savc_pair_rows(rows, "validation")

    assert len(leftovers) == 2
    assert len(pairs) == 8
    assert sum(row["target"] == "same_action" for row in pairs) == 4
    assert sum(row["target"] == "different_action" for row in pairs) == 4
    source_labels = {row["asset_id_sha256"]: row["target_label"] for row in rows}
    groups = {}
    for row in pairs:
        groups.setdefault(row["pair_group_sha256"], []).append(row)
    assert len(groups) == 1
    used_sources = set()
    for group in groups.values():
        assert len(group) == 8
        audio_labels = {}
        video_labels = {}
        for row in group:
            assert row["audio_asset_id_sha256"] != row["video_asset_id_sha256"]
            labels_match = (
                source_labels[row["audio_asset_id_sha256"]]
                == source_labels[row["video_asset_id_sha256"]]
            )
            assert labels_match == (row["target"] == "same_action")
            audio_labels.setdefault(row["audio_asset_id_sha256"], set()).add(row["target"])
            video_labels.setdefault(row["video_asset_id_sha256"], set()).add(row["target"])
            used_sources.add(row["audio_asset_id_sha256"])
            used_sources.add(row["video_asset_id_sha256"])
        assert all(
            labels == {"same_action", "different_action"} for labels in audio_labels.values()
        )
        assert all(
            labels == {"same_action", "different_action"} for labels in video_labels.values()
        )
    assert used_sources.isdisjoint(leftovers)
    assert used_sources | set(leftovers) == {row["asset_id_sha256"] for row in rows}


def test_savc_pairing_refuses_sealed_split_or_duplicate_source():
    with pytest.raises(ValueError, match="only accepts"):
        build_savc_pair_rows([], "sealed_audit")
    rows = [
        {"asset_id_sha256": "same", "feature_record": "a", "target_label": "a"},
        {"asset_id_sha256": "same", "feature_record": "b", "target_label": "b"},
    ]
    with pytest.raises(ValueError, match="duplicate source clip"):
        build_savc_pair_rows(rows, "train")


def test_savc_pair_groups_are_disjoint_and_deterministic():
    rows = [
        {
            "asset_id_sha256": f"asset-{index}",
            "feature_record": f"features/{index}.tfrecord",
            "target_label": label,
        }
        for index, label in enumerate(("a", "a", "a", "a", "b", "b", "b", "b"))
    ]
    first, leftovers = build_savc_pair_rows(rows, "train")
    second, second_leftovers = build_savc_pair_rows(rows, "train")

    assert first == second
    assert leftovers == second_leftovers == []
    grouped = {}
    for row in first:
        grouped.setdefault(row["pair_group_sha256"], []).append(row)
    assert len(grouped) == 2
    used_by_group = []
    for group in grouped.values():
        assert len(group) == 8
        used_by_group.append(
            {
                asset
                for row in group
                for asset in (row["audio_asset_id_sha256"], row["video_asset_id_sha256"])
            }
        )
    assert used_by_group[0].isdisjoint(used_by_group[1])


def test_canonical_jsonl_bytes_have_stable_lf_and_hash(tmp_path):
    rows = [{"z": "日本語", "a": 1}, {"a": 2}]
    payload = canonical_jsonl_bytes(rows)
    expected = b'{"a":1,"z":"\\u65e5\\u672c\\u8a9e"}\n{"a":2}\n'
    assert payload == expected
    assert b"\r\n" not in payload
    manifest = tmp_path / "stable.jsonl"
    manifest.write_bytes(payload)
    assert hashlib.sha256(manifest.read_bytes()).digest() == hashlib.sha256(payload).digest()
