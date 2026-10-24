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
            "frame_rate": 10.0,
            "audio_sample_rate": 100.0,
            "audio_samples": 2_000,
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
    records = list(source_candidate_records("v1", row(), cut_frame=-1))
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
    assert record["supervision"] == {
        "target_candidate_ids": [expected_target],
        "none_visible": False,
    }
    serialized_request = json.dumps(record["request"])
    assert "cup" not in serialized_request
    assert "opening" not in serialized_request
    assert "v1" not in serialized_request
    assert "object:9" not in serialized_request
    assert "target" not in serialized_request
    assert record["request"]["none_visible_option_id"] == "none_visible"
    assert record["group_id"] == "v1"
    assert record["split"] == "train"


def test_multiple_parents_are_preserved_without_single_winner_collapse():
    data = row()
    data["sound_localisation"][0]["parent_objects"] = [9, 2]
    distractor = dict(data["object_tracking"][1])
    distractor["id"] = 3
    distractor["label"] = "plate"
    data["object_tracking"].append(distractor)
    record = list(source_candidate_records("v1", data, cut_frame=-1))[0]
    assert len(record["supervision"]["target_candidate_ids"]) == 2
    assert len(set(record["supervision"]["target_candidate_ids"])) == 2


@pytest.mark.parametrize("split,video_id", [("valid", "v1"), ("train", "v2")])
def test_refuses_nontrain_or_mismatched_video(split, video_id):
    data = row()
    data["metadata"]["split"] = split
    data["metadata"]["video_id"] = video_id
    with pytest.raises(ValueError):
        list(source_candidate_records("v1", data, cut_frame=-1))


def test_invisible_source_has_explicit_none_visible_target():
    data = row()
    data["sound_localisation"][0]["is_visible"] = 0
    data["sound_localisation"][0]["parent_objects"] = [-1]
    record = list(source_candidate_records("v1", data, cut_frame=-1))[0]
    assert record["status"] == "candidate"
    assert record["supervision"] == {"target_candidate_ids": [], "none_visible": True}
    assert record["request"]["none_visible_option_id"] == "none_visible"


def test_cut_frame_is_mandatory_and_events_reaching_it_are_held():
    with pytest.raises(ValueError, match="explicit valid upstream cut frame"):
        list(source_candidate_records("v1", row()))
    data = row()
    data["sound_localisation"][0]["timestamps"] = [1_400_000, 1_600_000]
    data["sound_localisation"][0]["frame_ids"] = [14, 16]
    record = list(source_candidate_records("v1", data, cut_frame=15))[0]
    assert record["status"] == "held"
    assert "event_interval_exceeds_media_duration_or_cut" in record["hold_reasons"]
    assert "event_frame_range_reaches_answer_reveal_cut" in record["hold_reasons"]


def test_holds_unknown_missing_or_unobservable_parent_and_no_distractor():
    data = row()
    data["sound_localisation"][0]["parent_objects"] = [-1]
    assert list(source_candidate_records("v1", data, cut_frame=-1))[0]["status"] == "held"

    data = row()
    data["object_tracking"] = data["object_tracking"][:1]
    assert list(source_candidate_records("v1", data, cut_frame=-1))[0]["status"] == "held"


def test_holds_event_whose_timestamp_exceeds_video_duration():
    data = row()
    data["sound_localisation"][0]["timestamps"] = [100_000, 2_100_000]
    record = list(source_candidate_records("v1", data, cut_frame=-1))[0]
    assert record["status"] == "held"
    assert "event_interval_exceeds_media_duration_or_cut" in record["hold_reasons"]


def test_holds_event_whose_timestamp_exceeds_audio_duration_only():
    data = row()
    data["metadata"]["num_frames"] = 100
    data["metadata"]["audio_samples"] = 200
    data["sound_localisation"][0]["timestamps"] = [100_000, 2_100_000]
    record = list(source_candidate_records("v1", data, cut_frame=-1))[0]
    assert record["status"] == "held"
    assert "event_interval_exceeds_media_duration_or_cut" in record["hold_reasons"]

    data = row()
    data["object_tracking"] = [data["object_tracking"][0]]
    assert list(source_candidate_records("v1", data, cut_frame=-1))[0]["status"] == "held"


def test_frame_box_alignment_is_validated_and_masked_proposals_are_excluded():
    data = row()
    data["object_tracking"][1]["is_masked"] = [1, 1, 1]
    record = list(source_candidate_records("v1", data, cut_frame=-1))[0]
    assert record["status"] == "held"
    assert "no_non_source_visual_distractor" in record["hold_reasons"]

    data = row()
    data["object_tracking"][0]["bounding_boxes"].pop()
    with pytest.raises(ValueError):
        list(source_candidate_records("v1", data, cut_frame=-1))


def test_corpus_writer_is_deterministic_train_only_and_keeps_labels_out_of_request(
    tmp_path, monkeypatch
):
    source = tmp_path / "train.zip"
    cut_map_path = tmp_path / "cut_map.json"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": row()}))
    cut_map_path.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    monkeypatch.setattr(corpus_builder, "EXPECTED_TRAIN_SHA256", corpus_builder.sha256(source))
    monkeypatch.setattr(
        corpus_builder, "EXPECTED_CUT_MAP_SHA256", corpus_builder.sha256(cut_map_path)
    )
    first = corpus_builder.build(source, cut_map_path, tmp_path / "out-a")
    second = corpus_builder.build(source, cut_map_path, tmp_path / "out-b")
    assert first["row_count"] == 1
    assert first["candidate_examples"] == 1
    assert first["record_file_sha256"] == second["record_file_sha256"]
    assert first["validation_or_audit_loaded"] is False
    assert first["media_decoded"] is False
    records = (tmp_path / "out-a" / first["record_file"]).read_text(encoding="utf-8")
    assert '"label":"cup"' not in records
    assert '"label":"table"' not in records
    assert '"target_candidate_ids"' in records


def test_corpus_builder_coalesces_identical_model_inputs_with_same_targets(tmp_path, monkeypatch):
    source_row = row()
    duplicate = dict(source_row["sound_localisation"][0])
    duplicate["id"] = 5
    source_row["sound_localisation"].append(duplicate)
    source = tmp_path / "train.zip"
    cut_map_path = tmp_path / "cut_map.json"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": source_row}))
    cut_map_path.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    monkeypatch.setattr(corpus_builder, "EXPECTED_TRAIN_SHA256", corpus_builder.sha256(source))
    monkeypatch.setattr(
        corpus_builder, "EXPECTED_CUT_MAP_SHA256", corpus_builder.sha256(cut_map_path)
    )

    manifest = corpus_builder.build(source, cut_map_path, tmp_path / "out")
    records = [
        json.loads(line)
        for line in (tmp_path / "out" / manifest["record_file"]).read_text().splitlines()
    ]

    assert manifest["source_annotation_count"] == 2
    assert manifest["candidate_examples"] == 1
    assert manifest["duplicate_examples"] == 1
    assert manifest["pre_identity_status_counts"] == {"candidate": 2}
    assert [item["status"] for item in records] == ["candidate", "duplicate"]
    assert records[0]["provenance"]["coalesced_sound_annotation_ids"] == [4, 5]


def test_corpus_builder_holds_contradictory_identical_interval_inputs(tmp_path, monkeypatch):
    source_row = row()
    invisible = dict(source_row["sound_localisation"][0])
    invisible.update(
        {
            "id": 5,
            "label_id": 13,
            "label": "Other:background sound",
            "is_visible": 0,
            "parent_objects": [-1],
        }
    )
    source_row["sound_localisation"].append(invisible)
    source = tmp_path / "train.zip"
    cut_map_path = tmp_path / "cut_map.json"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": source_row}))
    cut_map_path.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    monkeypatch.setattr(corpus_builder, "EXPECTED_TRAIN_SHA256", corpus_builder.sha256(source))
    monkeypatch.setattr(
        corpus_builder, "EXPECTED_CUT_MAP_SHA256", corpus_builder.sha256(cut_map_path)
    )

    manifest = corpus_builder.build(source, cut_map_path, tmp_path / "out")
    records = [
        json.loads(line)
        for line in (tmp_path / "out" / manifest["record_file"]).read_text().splitlines()
    ]

    assert manifest["candidate_examples"] == 0
    assert manifest["status_counts"]["held"] == 2
    assert all("contradictory_identical_model_input" in item["hold_reasons"] for item in records)


def test_corpus_writer_rejects_unpinned_archive_hash(tmp_path, monkeypatch):
    source = tmp_path / "train.zip"
    cut_map_path = tmp_path / "cut_map.json"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("all_train.json", json.dumps({"v1": row()}))
    cut_map_path.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    monkeypatch.setattr(
        corpus_builder, "EXPECTED_CUT_MAP_SHA256", corpus_builder.sha256(cut_map_path)
    )
    with pytest.raises(ValueError, match="archive hash differs"):
        corpus_builder.build(source, cut_map_path, tmp_path / "out")
