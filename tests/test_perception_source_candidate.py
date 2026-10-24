import hashlib
import json
import zipfile

import pytest

from scripts import build_perception_source_candidate_corpus as corpus_builder
from tiny_omni_decision.perception_source_candidate import source_candidate_records


def row():
    return {
        "metadata": {
            "split": "train",
            "video_id": "v1",
            "num_frames": 20,
            "is_cup_game": 0,
        },
        "sound_localisation": [
            {
                "id": 4,
                "label_id": 2,
                "label": "Object:opening",
                "timestamps": [100_000, 300_000],
                "frame_ids": [1, 3],
                "parent_objects": [9],
                "is_visible": 1,
            }
        ],
        "object_tracking": [
            {
                "id": 9,
                "label": "cup",
                "frame_ids": [0, 1, 2, 3, 4],
                "bounding_boxes": [[0.1, 0.1, 0.4, 0.4]] * 5,
                "is_masked": [0] * 5,
            },
            {
                "id": 2,
                "label": "table",
                "frame_ids": [1, 2, 3],
                "bounding_boxes": [[0.5, 0.5, 0.8, 0.8]] * 3,
                "is_masked": [0] * 3,
            },
        ],
    }


def test_record_has_hidden_multi_candidate_targets_and_only_train_identity():
    records = list(source_candidate_records("v1", row()))
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "candidate"
    assert record["request"]["task"] == "audio_visual_source_candidate_reranking"
    assert {candidate["candidate_id"] for candidate in record["request"]["candidates"]} == {
        "option_0",
        "option_1",
    }
    hashed_order = sorted(
        (hashlib.sha256(f"v1:4:{object_id}".encode()).hexdigest(), object_id)
        for object_id in (2, 9)
    )
    target_index = next(
        index for index, (_, object_id) in enumerate(hashed_order) if object_id == 9
    )
    expected_target = f"option_{target_index}"
    assert record["supervision"] == {"target_candidate_ids": [expected_target]}
    serialized_request = json.dumps(record["request"])
    assert "cup" not in serialized_request
    assert "opening" not in serialized_request
    assert "v1" not in serialized_request
    assert "object:9" not in serialized_request
    assert "target" not in serialized_request
    assert record["group_id"] == "v1"
    assert record["split"] == "train"


def test_multiple_parents_are_preserved_without_single_winner_collapse():
    data = row()
    data["sound_localisation"][0]["parent_objects"] = [9, 2]
    distractor = dict(data["object_tracking"][1])
    distractor["id"] = 3
    distractor["label"] = "plate"
    data["object_tracking"].append(distractor)
    record = list(source_candidate_records("v1", data))[0]
    assert len(record["supervision"]["target_candidate_ids"]) == 2
    assert len(set(record["supervision"]["target_candidate_ids"])) == 2


@pytest.mark.parametrize("split,video_id", [("valid", "v1"), ("train", "v2")])
def test_refuses_nontrain_or_mismatched_video(split, video_id):
    data = row()
    data["metadata"]["split"] = split
    data["metadata"]["video_id"] = video_id
    with pytest.raises(ValueError):
        list(source_candidate_records("v1", data))


def test_holds_unknown_missing_or_unobservable_parent_and_no_distractor():
    data = row()
    data["sound_localisation"][0]["parent_objects"] = [-1]
    assert list(source_candidate_records("v1", data))[0]["status"] == "held"

    data = row()
    data["object_tracking"] = data["object_tracking"][:1]
    assert list(source_candidate_records("v1", data))[0]["status"] == "held"

    data = row()
    data["object_tracking"] = [data["object_tracking"][0]]
    assert list(source_candidate_records("v1", data))[0]["status"] == "held"


def test_frame_box_alignment_is_validated_and_masked_proposals_are_excluded():
    data = row()
    data["object_tracking"][1]["is_masked"] = [1, 1, 1]
    record = list(source_candidate_records("v1", data))[0]
    assert record["status"] == "held"
    assert "no_non_source_visual_distractor" in record["hold_reasons"]

    data = row()
    data["object_tracking"][0]["bounding_boxes"].pop()
    with pytest.raises(ValueError):
        list(source_candidate_records("v1", data))


def test_corpus_writer_is_deterministic_train_only_and_keeps_labels_out_of_request(
    tmp_path, monkeypatch
):
    source = tmp_path / "train.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": row()}))
    monkeypatch.setattr(corpus_builder, "EXPECTED_TRAIN_SHA256", corpus_builder.sha256(source))
    first = corpus_builder.build(source, tmp_path / "out-a")
    second = corpus_builder.build(source, tmp_path / "out-b")
    assert first["row_count"] == 1
    assert first["candidate_examples"] == 1
    assert first["record_file_sha256"] == second["record_file_sha256"]
    assert first["validation_or_audit_loaded"] is False
    assert first["media_decoded"] is False
    records = (tmp_path / "out-a" / first["record_file"]).read_text(encoding="utf-8")
    assert '"label":"cup"' not in records
    assert '"label":"table"' not in records
    assert '"target_candidate_ids"' in records


def test_corpus_writer_rejects_unpinned_archive_hash(tmp_path):
    source = tmp_path / "train.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": row()}))
    with pytest.raises(ValueError, match="archive hash differs"):
        corpus_builder.build(source, tmp_path / "out")
