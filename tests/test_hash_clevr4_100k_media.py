import json
import zipfile

from scripts.hash_clevr4_100k_media import hash_indexed_media


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hashes_only_indexed_rows_and_checks_cross_split_and_prior_overlap(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("images/train.png", b"train bytes")
        zipped.writestr("images/validation.png", b"validation bytes")
        # This held-out source member is deliberately not valid media.
        zipped.writestr("images/official-val.png", b"not opened")
    index = tmp_path / "index"
    index.mkdir()
    _write_rows(
        index / "train-media-manifest.jsonl",
        [{"id": "train", "archive_member": "images/train.png", "media_sha256": None}],
    )
    _write_rows(
        index / "validation-media-manifest.jsonl",
        [{"id": "validation", "archive_member": "images/validation.png", "media_sha256": None}],
    )
    previous = tmp_path / "previous.jsonl"
    _write_rows(previous, [{"sha256": "00" * 32}])

    result = hash_indexed_media(archive, index, previous, tmp_path / "hashed")

    assert result["indexed_train_examples"] == 1
    assert result["indexed_validation_examples"] == 1
    assert result["train_validation_media_overlap"] == 0
    assert result["overlap_with_previous_train_media"] == 0
    assert result["official_val_media_opened"] == 0
    assert result["files_extracted"] == 0
    train = [
        json.loads(line)
        for line in (tmp_path / "hashed" / "train-media-manifest.jsonl").read_text().splitlines()
    ]
    assert train[0]["media_sha256"]
