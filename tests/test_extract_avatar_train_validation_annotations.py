from __future__ import annotations

import json
import zipfile

import pytest

from scripts.extract_avatar_train_validation_annotations import extract_annotations
from scripts.prepare_avatar_group_split import build_manifest


def _video_archive(path, source_ids):
    with zipfile.ZipFile(path, "w") as archive:
        for index, source_id in enumerate(source_ids):
            archive.writestr(f"video/{source_id}_{index:05d}.mp4", b"x")
    return path


def _metadata_archive(path, clip_ids):
    with zipfile.ZipFile(path, "w") as archive:
        for index, clip_id in enumerate(clip_ids):
            archive.writestr(f"{clip_id}/{index:05d}.json", json.dumps({"label": index}))
            archive.writestr(f"{clip_id}/{index:05d}.jpg", b"not extracted")
    return path


def test_extracts_only_train_and_validation_and_never_opens_audit(tmp_path, monkeypatch):
    video = _video_archive(tmp_path / "video.zip", [f"{i:011d}" for i in range(30)])
    split = build_manifest(video, check_pin=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    audit_ids = set(split["splits"]["audit"]["source_video_ids"])
    all_ids = sorted(
        audit_ids
        | set(split["splits"]["train"]["source_video_ids"])
        | set(split["splits"]["validation"]["source_video_ids"])
    )
    clip_ids = [f"{source_id}_{index:05d}" for index, source_id in enumerate(all_ids)]
    metadata = _metadata_archive(tmp_path / "metadata.zip", clip_ids)

    original_open = zipfile.ZipFile.open
    opened = []

    def tracked_open(bundle, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        opened.append(filename)
        if filename.split("/", 1)[0].split("_", 1)[0] in audit_ids:
            raise AssertionError("audit annotation member must not be opened")
        return original_open(bundle, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", tracked_open)
    output = tmp_path / "selected"
    receipt = extract_annotations(metadata, split_path, output, check_pins=False)

    assert receipt["audit_annotation_members_opened"] is False
    assert receipt["withheld_audit_annotation_json_count"] == len(audit_ids)
    assert receipt["extracted_train_validation_json_count"] == 27
    assert len(opened) == 27
    assert not (output / "audit").exists()
    assert len(list(output.rglob("*.jpg"))) == 0
    assert len(list(output.rglob("*.json"))) == 28  # extracted-files manifest + selected frames


def test_refuses_to_overwrite_existing_output_directory(tmp_path):
    video = _video_archive(tmp_path / "video.zip", [f"{i:011d}" for i in range(30)])
    split = build_manifest(video, check_pin=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    clip_ids = [f"{i:011d}_{i:05d}" for i in range(30)]
    metadata = _metadata_archive(tmp_path / "metadata.zip", clip_ids)
    output = tmp_path / "selected"
    output.mkdir()

    with pytest.raises(FileExistsError, match="new output directory required"):
        extract_annotations(metadata, split_path, output, check_pins=False)
