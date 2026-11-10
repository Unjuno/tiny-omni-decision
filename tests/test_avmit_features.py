from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tiny_omni_decision.avmit_features import audit_and_cache_avmit_features
from tiny_omni_decision.avmit_tfrecord import _masked_crc32c


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _feature(value: bytes) -> bytes:
    return _bytes_field(1, _bytes_field(1, value))


def _map_entry(key: str, value: bytes) -> bytes:
    return _bytes_field(1, key.encode()) + _bytes_field(2, value)


def _sequence_example(filename: str, seed: int) -> bytes:
    context = _bytes_field(1, _map_entry("filename", _feature(filename.encode())))
    audio = [np.array([seed, seed + 0.5], dtype="<f4").tobytes()]
    video = [np.array([seed, seed + 1, seed + 2], dtype="<f4").tobytes()]
    feature_lists = _bytes_field(
        1,
        _map_entry("audio", b"".join(_bytes_field(1, _feature(value)) for value in audio)),
    ) + _bytes_field(
        1,
        _map_entry(
            "visual_object",
            b"".join(_bytes_field(1, _feature(value)) for value in video),
        ),
    )
    return _bytes_field(1, context) + _bytes_field(2, feature_lists)


def _tfrecord(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", _masked_crc32c(length))
        + payload
        + struct.pack("<I", _masked_crc32c(payload))
    )


def _write_pair_manifests(directory):
    directory.mkdir()
    source_rows = []
    for split in ("train", "calibration", "validation"):
        source_names = [
            f"class{'a' if index < 2 else 'b'}/{split}-{index}-long-name.mp4" for index in range(4)
        ]
        assets = [hashlib.sha256(name.casefold().encode()).hexdigest() for name in source_names]
        records = [
            f"class{'a' if index < 2 else 'b'}/{split}-{index}~1.tfrecord" for index in range(4)
        ]
        source_rows.extend(
            {"filename": source_names[index], "tfrecord_filename": records[index]}
            for index in range(4)
        )
        rows = []
        group = hashlib.sha256(f"group-{split}".encode()).hexdigest()
        entries = (
            (0, 1, "same_action"),
            (1, 0, "same_action"),
            (2, 3, "same_action"),
            (3, 2, "same_action"),
            (0, 2, "different_action"),
            (2, 1, "different_action"),
            (1, 3, "different_action"),
            (3, 0, "different_action"),
        )
        for audio, video, target in entries:
            rows.append(
                {
                    "split": split,
                    "pair_group_sha256": group,
                    "audio_asset_id_sha256": assets[audio],
                    "video_asset_id_sha256": assets[video],
                    "audio_feature_record": records[audio],
                    "video_feature_record": records[video],
                    "target": target,
                }
            )
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        (directory / f"{split}.jsonl").write_text(payload, encoding="utf-8")
    # If the cache code accidentally opens this file, parsing fails loudly.
    (directory / "sealed_audit.jsonl").write_text("not-json; never read\n", encoding="utf-8")
    with (directory / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "tfrecord_filename"))
        writer.writeheader()
        writer.writerows(source_rows)


def _write_archive(path, pair_dir):
    with (pair_dir / "ratings.csv").open(encoding="utf-8", newline="") as handle:
        source_by_alias = {
            row["tfrecord_filename"].casefold(): row["filename"] for row in csv.DictReader(handle)
        }
    with tarfile.open(path, "w:") as archive:
        for _alias, source_name in source_by_alias.items():
            record = str(Path(source_name).with_suffix(".tfrecord")).replace("\\", "/")
            payload = gzip.compress(
                _tfrecord(_sequence_example(source_name, len(archive.getmembers()) + 1))
            )
            member = tarfile.TarInfo(f"release/features/{record}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(path, "a:") as archive:
        decoy = tarfile.TarInfo("release/features/wrong-class/train-0-long-name.tfrecord")
        decoy_payload = b"same basename, wrong class path"
        decoy.size = len(decoy_payload)
        archive.addfile(decoy, io.BytesIO(decoy_payload))
        sealed = tarfile.TarInfo("release/features/sealed/unopened.tfrecord")
        sealed_payload = b"invalid-tfrecord!!"
        sealed.size = len(sealed_payload)
        archive.addfile(sealed, io.BytesIO(sealed_payload))


def test_avmit_cache_reads_only_allowlisted_nonsealed_records(tmp_path):
    pair_dir = tmp_path / "pairs-v2"
    _write_pair_manifests(pair_dir)
    archive_path = tmp_path / "features.tar"
    _write_archive(archive_path, pair_dir)
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    output = tmp_path / "cache"

    report = audit_and_cache_avmit_features(
        archive_path,
        pair_dir / "ratings.csv",
        pair_dir,
        output,
        expected_archive_sha256=archive_sha,
        expected_source_metadata_sha256=hashlib.sha256(
            (pair_dir / "ratings.csv").read_bytes()
        ).hexdigest(),
        expected_archive_size=archive_path.stat().st_size,
        expected_archive_md5=archive_md5,
    )

    assert report["sealed_payloads_opened"] == 0
    assert report["raw_media_opened"] == 0
    for split in ("train", "calibration", "validation"):
        split_info = report["feature_cache"][split]
        assert split_info["unique_source_clips"] == 4
        assert split_info["pair_manifest_rows"] == 8
        assert split_info["audio_width"] == 2
        assert split_info["video_width"] == 3
        index = [
            json.loads(line)
            for line in (output / split / "index.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(index) == 4
        assert all("target" not in row for row in index)
    all_output = b"".join(path.read_bytes() for path in output.rglob("*.*"))
    assert b"sealed/unopened" not in all_output
    assert b"invalid-tfrecord!!" not in all_output
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_and_cache_avmit_features(
            archive_path,
            pair_dir / "ratings.csv",
            pair_dir,
            output,
            expected_archive_sha256=archive_sha,
            expected_source_metadata_sha256=hashlib.sha256(
                (pair_dir / "ratings.csv").read_bytes()
            ).hexdigest(),
            expected_archive_size=archive_path.stat().st_size,
            expected_archive_md5=archive_md5,
        )


def test_avmit_cache_fails_closed_on_archive_hash_mismatch(tmp_path):
    pair_dir = tmp_path / "pairs-v2"
    _write_pair_manifests(pair_dir)
    archive_path = tmp_path / "features.tar"
    _write_archive(archive_path, pair_dir)
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_and_cache_avmit_features(
            archive_path,
            pair_dir / "ratings.csv",
            pair_dir,
            output,
            expected_archive_sha256="0" * 64,
            expected_source_metadata_sha256=hashlib.sha256(
                (pair_dir / "ratings.csv").read_bytes()
            ).hexdigest(),
            expected_archive_size=archive_path.stat().st_size,
            expected_archive_md5=hashlib.md5(archive_path.read_bytes()).hexdigest(),
        )
    assert not output.exists()


def test_avmit_cache_rejects_cross_split_asset_overlap(tmp_path):
    pair_dir = tmp_path / "pairs-v2"
    _write_pair_manifests(pair_dir)
    train = pair_dir / "train.jsonl"
    validation = pair_dir / "validation.jsonl"
    train_row = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
    validation_text = validation.read_text(encoding="utf-8")
    validation_row = json.loads(validation_text.splitlines()[0])
    original_asset = validation_row["audio_asset_id_sha256"]
    validation.write_text(
        validation_text.replace(original_asset, train_row["audio_asset_id_sha256"]),
        encoding="utf-8",
    )
    archive_path = tmp_path / "features.tar"
    _write_archive(archive_path, pair_dir)
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="source asset appears in more than one split"):
        audit_and_cache_avmit_features(
            archive_path,
            pair_dir / "ratings.csv",
            pair_dir,
            output,
            expected_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            expected_source_metadata_sha256=hashlib.sha256(
                (pair_dir / "ratings.csv").read_bytes()
            ).hexdigest(),
            expected_archive_size=archive_path.stat().st_size,
            expected_archive_md5=hashlib.md5(archive_path.read_bytes()).hexdigest(),
        )
    assert not output.exists()
