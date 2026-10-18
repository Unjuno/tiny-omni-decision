from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.index_clevr4_100k import build_index


def test_index_reads_only_official_train_labels_and_does_not_open_test_media(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "fixture.zip"
    output = tmp_path / "indexed"
    annotation = (
        b'{"train-a":{"split":"train","color":"red","texture":"rubber",'
        b'"count":1,"shape":"cube"},'
        b'"train-b":{"split":"train","color":"blue","texture":"metal",'
        b'"count":2,"shape":"sphere"},'
        b'"train-c":{"split":"train","color":"green","texture":"wave",'
        b'"count":3,"shape":"cone"},'
        b'"train-d":{"split":"train","color":"yellow","texture":"brick",'
        b'"count":4,"shape":"torus"},'
        b'"test-only":{"color":unopened_test_label,"split":"test"}}'
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("clevr_4_annots.json", annotation)
        for image_id in ("train-a", "train-b", "train-c", "train-d", "test-only"):
            zipped.writestr(f"images/{image_id}.png", b"synthetic-image-bytes")

    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo("images/test-only.png")
        with archive.open("r+b") as raw:
            raw.seek(info.header_offset)
            header = raw.read(30)
            filename_length = int.from_bytes(header[26:28], "little")
            extra_length = int.from_bytes(header[28:30], "little")
            payload_offset = info.header_offset + 30 + filename_length + extra_length
            raw.seek(payload_offset)
            original = raw.read(1)
            raw.seek(payload_offset)
            raw.write(bytes([original[0] ^ 0xFF]))

    report = build_index(
        archive,
        output,
        expected_archive_bytes=archive.stat().st_size,
        expected_sha512=hashlib.sha512(archive.read_bytes()).hexdigest(),
        expected_image_count=5,
        validation_fraction=0.25,
        seed=17,
    )

    assert report["official_split_counts"] == {"train": 4, "test": 1}
    assert report["official_test_annotation_values_decoded"] == 0
    assert report["official_test_image_members_opened"] == 0
    assert report["train_validation_asset_overlap"] == 0
    assert report["train_examples"] == 3
    assert report["validation_examples"] == 1
    assert report["media_hashes_computed"] is False

    train_rows = [
        json.loads(line) for line in (output / "train-labels.jsonl").read_text().splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (output / "validation-labels.jsonl").read_text().splitlines()
    ]
    assert len(train_rows) + len(validation_rows) == 4
    assert "test-only" not in {row["id"] for row in train_rows + validation_rows}
