import hashlib
import json

import pytest

from scripts.prepare_clevrer_synthetic_av_v2_feature_input import prepare_feature_input


def _manifest(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        items = [row for row in rows if row["split"] == split]
        (path / f"{split}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
        )
    (path / "fetch-report.json").write_text("{}", encoding="utf-8")


def test_prepares_absolute_paths_and_rehashes_media(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "media").mkdir(parents=True)
    (new_root / "media").mkdir(parents=True)
    old_media = old_root / "media" / "old.mp4"
    new_media = new_root / "media" / "new.mp4"
    old_media.write_bytes(b"old media")
    new_media.write_bytes(b"new media")
    old_hash = hashlib.sha256(old_media.read_bytes()).hexdigest()
    new_hash = hashlib.sha256(new_media.read_bytes()).hexdigest()
    _manifest(
        old_root,
        [
            {"id": "old-train", "split": "train", "media_sha256": old_hash},
            {"id": "old-val", "split": "validation", "media_sha256": old_hash},
        ],
    )
    _manifest(
        tmp_path / "samples",
        [
            {
                "id": "old-train",
                "split": "train",
                "media_sha256": old_hash,
                "media_path": "media/old.mp4",
            },
            {
                "id": "new-val",
                "split": "validation",
                "media_sha256": new_hash,
                "media_path": "media/new.mp4",
            },
        ],
    )

    report = prepare_feature_input(
        sample_dir=tmp_path / "samples",
        old_sample_dir=old_root,
        expanded_media_root=new_root,
        output_dir=tmp_path / "prepared",
    )
    train_row = json.loads((tmp_path / "prepared" / "train.jsonl").read_text())
    val_row = json.loads((tmp_path / "prepared" / "validation.jsonl").read_text())
    assert train_row["media_path"] == str(old_media.resolve())
    assert val_row["media_path"] == str(new_media.resolve())
    assert report["unique_media_count"] == 2
    with pytest.raises(FileExistsError):
        prepare_feature_input(
            sample_dir=tmp_path / "samples",
            old_sample_dir=old_root,
            expanded_media_root=new_root,
            output_dir=tmp_path / "prepared",
        )


def test_rejects_media_content_hash_mismatch(tmp_path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "media").mkdir(parents=True)
    (new_root / "media").mkdir(parents=True)
    stale = hashlib.sha256(b"original").hexdigest()
    (old_root / "media" / "old.mp4").write_bytes(b"changed")
    (new_root / "media" / "new.mp4").write_bytes(b"new")
    _manifest(
        old_root,
        [{"id": "old", "split": "train", "media_sha256": stale}],
    )
    new_hash = hashlib.sha256(b"new").hexdigest()
    _manifest(
        tmp_path / "samples",
        [
            {
                "id": "old",
                "split": "train",
                "media_sha256": stale,
                "media_path": "media/old.mp4",
            },
            {
                "id": "new",
                "split": "validation",
                "media_sha256": new_hash,
                "media_path": "media/new.mp4",
            },
        ],
    )
    with pytest.raises(ValueError, match="media identity mismatch"):
        prepare_feature_input(
            sample_dir=tmp_path / "samples",
            old_sample_dir=old_root,
            expanded_media_root=new_root,
            output_dir=tmp_path / "prepared",
        )
