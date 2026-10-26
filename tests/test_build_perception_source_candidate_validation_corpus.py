import json
import zipfile

import pytest

from scripts import build_perception_source_candidate_validation_corpus as builder


def validation_row(video_id="v1", *, split="valid"):
    return {
        "metadata": {
            "split": split,
            "video_id": video_id,
            "num_frames": 20,
            "frame_rate": 10.0,
            "audio_sample_rate": 100.0,
            "audio_samples": 2_000,
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


def test_builder_freezes_exact_validation_events_and_preserves_source_split(tmp_path, monkeypatch):
    source_zip = tmp_path / "validation.zip"
    cut_map = tmp_path / "cut-map.json"
    selection = tmp_path / "selection.json"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "all_valid.json",
            json.dumps({"v1": validation_row()}),
        )
    cut_map.write_text(
        json.dumps({"nid": None, "v1": -1, "other_valid_task_video": 3}),
        encoding="utf-8",
    )
    selection.write_text(
        json.dumps(
            {
                "source_split": "official validation only",
                "official_test_split_opened": False,
                "selected_video_count": 1,
                "events": [
                    {
                        "event_id": 4,
                        "video_id": "v1",
                        "split": "valid",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_SHA256", builder.sha256(source_zip))
    monkeypatch.setattr(builder, "EXPECTED_CUT_MAP_SHA256", builder.sha256(cut_map))

    manifest = builder.build(
        source_zip, cut_map, selection, tmp_path / "transfer-corpus"
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "transfer-corpus" / manifest["record_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["source_split"] == "official_validation"
    assert manifest["official_test_split_opened"] is False
    assert manifest["selected_event_count"] == 1
    assert rows[0]["record_id"] == "v1:sound:4"
    assert rows[0]["split"] == "validation"
    assert rows[0]["provenance"]["source_split"] == "official_validation"


def test_builder_rejects_selection_with_duplicate_videos(tmp_path, monkeypatch):
    source_zip = tmp_path / "validation.zip"
    cut_map = tmp_path / "cut-map.json"
    selection = tmp_path / "selection.json"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "all_valid.json",
            json.dumps({"v1": validation_row()}),
        )
    cut_map.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    selection.write_text(
        json.dumps(
            {
                "source_split": "official validation only",
                "official_test_split_opened": False,
                "selected_video_count": 2,
                "events": [
                    {"event_id": 4, "video_id": "v1", "split": "valid"},
                    {"event_id": 5, "video_id": "v1", "split": "valid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_SHA256", builder.sha256(source_zip))
    monkeypatch.setattr(builder, "EXPECTED_CUT_MAP_SHA256", builder.sha256(cut_map))

    with pytest.raises(ValueError, match="one event per validation video"):
        builder.build(source_zip, cut_map, selection, tmp_path / "transfer-corpus")


def test_builder_accepts_reused_local_sound_ids_across_video_groups(tmp_path, monkeypatch):
    source_zip = tmp_path / "validation.zip"
    cut_map = tmp_path / "cut-map.json"
    selection = tmp_path / "selection.json"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "all_valid.json",
            json.dumps({"v1": validation_row("v1"), "v2": validation_row("v2")}),
        )
    cut_map.write_text(json.dumps({"nid": -1, "v1": -1, "v2": -1}), encoding="utf-8")
    selection.write_text(
        json.dumps(
            {
                "source_split": "official validation only",
                "official_test_split_opened": False,
                "selected_video_count": 2,
                "events": [
                    {"event_id": 4, "video_id": "v1", "split": "valid"},
                    {"event_id": 4, "video_id": "v2", "split": "valid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_SHA256", builder.sha256(source_zip))
    monkeypatch.setattr(builder, "EXPECTED_CUT_MAP_SHA256", builder.sha256(cut_map))

    manifest = builder.build(
        source_zip, cut_map, selection, tmp_path / "transfer-corpus"
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "transfer-corpus" / manifest["record_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["v1:sound:4", "v2:sound:4"]


def test_builder_fails_closed_when_validation_rows_lack_object_tracks(
    tmp_path, monkeypatch
):
    source_zip = tmp_path / "validation.zip"
    cut_map = tmp_path / "cut-map.json"
    selection = tmp_path / "selection.json"
    row = validation_row()
    row.pop("object_tracking")
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("all_valid.json", json.dumps({"v1": row}))
    cut_map.write_text(json.dumps({"nid": -1, "v1": -1}), encoding="utf-8")
    selection.write_text(
        json.dumps(
            {
                "source_split": "official validation only",
                "official_test_split_opened": False,
                "selected_video_count": 1,
                "events": [{"event_id": 4, "video_id": "v1", "split": "valid"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_VALIDATION_SHA256", builder.sha256(source_zip))
    monkeypatch.setattr(builder, "EXPECTED_CUT_MAP_SHA256", builder.sha256(cut_map))

    with pytest.raises(ValueError, match="object_tracking annotations"):
        builder.build(source_zip, cut_map, selection, tmp_path / "transfer-corpus")
