from __future__ import annotations

import io
import json
import zipfile

import pytest

from scripts.fetch_clevrer_synthetic_av_annotations_v2 import inspect_annotation_archive


def _annotation(scene: int) -> bytes:
    return json.dumps(
        {
            "scene_index": scene,
            "video_filename": f"video_{scene:05d}.mp4",
            "object_property": [
                {"object_id": 0, "color": "blue", "material": "metal", "shape": "cube"},
                {"object_id": 1, "color": "red", "material": "rubber", "shape": "sphere"},
            ],
            "collision": [{"object_ids": [0, 1], "frame_id": 32}],
        }
    ).encode()


def _zip_archive(*scenes: int) -> io.BytesIO:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene in scenes:
            bucket_start = scene // 1000 * 1000
            archive.writestr(
                f"annotation_{bucket_start:05d}-{bucket_start + 1000:05d}/"
                f"annotation_{scene:05d}.json",
                _annotation(scene),
            )
    output.seek(0)
    return output


def test_extracts_only_requested_train_scene_members_and_counts_eligible_events() -> None:
    payloads, report = inspect_annotation_archive(_zip_archive(1, 2, 3), [1, 2], [3])
    assert sorted(payloads) == [
        "annotation_00001.json",
        "annotation_00002.json",
        "annotation_00003.json",
    ]
    assert report["eligible_scene_counts"] == {"train": 2, "validation": 1}
    assert report["eligible_collision_counts"] == {"train": 2, "validation": 1}
    assert report["archive_member_crc_checks"] == 3


def test_rejects_annotation_scene_identity_mismatch() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("annotation_00000-01000/annotation_00001.json", _annotation(2))
    archive.seek(0)
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_annotation_archive(archive, [1], [3])


def test_rejects_missing_or_duplicate_archive_member() -> None:
    with pytest.raises(ValueError, match="found 0"):
        inspect_annotation_archive(_zip_archive(1), [1], [3])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("a/annotation_00001.json", _annotation(1))
        output.writestr("b/annotation_00001.json", _annotation(1))
        output.writestr("a/annotation_00003.json", _annotation(3))
    archive.seek(0)
    with pytest.raises(ValueError, match="found 2"):
        inspect_annotation_archive(archive, [1], [3])
