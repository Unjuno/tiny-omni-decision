import hashlib
import io
import json

import pytest

from scripts.materialize_clevrer_temporal_media import materialize_media


def test_materialize_media_hashes_assets_and_preserves_scene_split(tmp_path):
    rows = {
        "train": [
            {
                "id": "clevrer-train-12345-q000",
                "split": "train",
                "scene_index": 12345,
                "video_filename": "video_12345.mp4",
                "question": "when is the event?",
                "options": ["before", "after"],
            }
        ],
        "validation": [
            {
                "id": "clevrer-validation-12346-q000",
                "split": "validation",
                "scene_index": 12346,
                "video_filename": "video_12346.mp4",
                "question": "what color is the object?",
                "options": ["red", "blue"],
            }
        ],
    }

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def namelist(self):
            return ["videos/video_12345.mp4", "videos/video_12346.mp4"]

        def open(self, member):
            return io.BytesIO(b"\x00\x00\x00\x18ftypisom-" + member.encode())

    output = tmp_path / "sample"
    result = materialize_media(
        rows,
        Archive(),
        output,
        revision="test-revision",
        archive_opener=Archive,
        worker_count=2,
    )

    assert result["scene_overlap"] == 0
    assert result["media_hash_overlap"] == 0
    assert result["worker_count"] == 2
    report = json.loads((output / "fetch-report.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "pretraining-gate.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete_temporal_media_sample"
    assert gate["status"] == "pretraining_gate_pass"
    for split in ("train", "validation"):
        row = json.loads((output / f"{split}.jsonl").read_text(encoding="utf-8"))
        path = output / row["media_path"]
        assert path.is_file()
        assert row["split"] == split
        assert row["scene_group_id"] == f"clevrer:{split}:{row['scene_index']}"
        assert row["media_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_media_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_media({"train": [], "validation": []}, None, output, revision="test")


def test_materialize_media_rejects_scene_overlap_and_invalid_media(tmp_path):
    rows = {
        "train": [{"id": "a", "split": "train", "scene_index": 1, "video_filename": "x.mp4"}],
        "validation": [
            {"id": "b", "split": "validation", "scene_index": 1, "video_filename": "y.mp4"}
        ],
    }
    with pytest.raises(ValueError, match="scene overlap"):
        materialize_media(rows, None, tmp_path / "overlap", revision="test")

    rows["validation"][0]["scene_index"] = 2
    rows["train"][0].update({"question": "same content", "options": ["a", "b"]})
    rows["validation"][0].update({"question": "same content", "options": ["b", "a"]})
    with pytest.raises(ValueError, match="normalized question/options overlap"):
        materialize_media(rows, None, tmp_path / "content-overlap", revision="test")

    rows["validation"][0].update({"question": "different", "options": ["a", "b"]})

    class InvalidArchive:
        def namelist(self):
            return ["x.mp4", "y.mp4"]

        def open(self, _member):
            return io.BytesIO(b"not an mp4")

    with pytest.raises(ValueError, match="invalid MP4"):
        materialize_media(rows, InvalidArchive(), tmp_path / "invalid", revision="test")
