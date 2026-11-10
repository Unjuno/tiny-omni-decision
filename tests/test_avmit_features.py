from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile

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
    for split in ("train", "calibration", "validation"):
        assets = [hashlib.sha256(f"{split}-{index}".encode()).hexdigest() for index in range(4)]
        records = [
            f"class{'a' if index < 2 else 'b'}/{split}-{index}.tfrecord" for index in range(4)
        ]
        rows = []
        group = hashlib.sha256(f"group-{split}".encode()).hexdigest()
        entries = (
            (0, 1, "same_action"),
            (1, 0, "same_action"),
            (2, 3, "same_action"),
            (3, 2, "same_action"),
            (0, 2, "different_action"),
            (2, 0, "different_action"),
            (1, 3, "different_action"),
            (3, 1, "different_action"),
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


def _write_archive(path):
    with tarfile.open(path, "w:") as archive:
        for split in ("train", "calibration", "validation"):
            for index in range(4):
                label = "classa" if index < 2 else "classb"
                record = f"{label}/{split}-{index}.tfrecord"
                payload = _tfrecord(
                    _sequence_example(record.removesuffix(".tfrecord") + ".mp4", index + 1)
                )
                member = tarfile.TarInfo(f"release/features/{record}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        sealed = tarfile.TarInfo("release/features/sealed/unopened.tfrecord")
        sealed_payload = b"invalid-tfrecord!!"
        sealed.size = len(sealed_payload)
        archive.addfile(sealed, io.BytesIO(sealed_payload))


def test_avmit_cache_reads_only_allowlisted_nonsealed_records(tmp_path):
    pair_dir = tmp_path / "pairs-v2"
    _write_pair_manifests(pair_dir)
    archive_path = tmp_path / "features.tar"
    _write_archive(archive_path)
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    output = tmp_path / "cache"

    report = audit_and_cache_avmit_features(
        archive_path,
        pair_dir,
        output,
        expected_archive_sha256=archive_sha,
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
            pair_dir,
            output,
            expected_archive_sha256=archive_sha,
            expected_archive_size=archive_path.stat().st_size,
            expected_archive_md5=archive_md5,
        )


def test_avmit_cache_fails_closed_on_archive_hash_mismatch(tmp_path):
    pair_dir = tmp_path / "pairs-v2"
    _write_pair_manifests(pair_dir)
    archive_path = tmp_path / "features.tar"
    _write_archive(archive_path)
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_and_cache_avmit_features(
            archive_path,
            pair_dir,
            output,
            expected_archive_sha256="0" * 64,
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
    _write_archive(archive_path)
    output = tmp_path / "cache"

    with pytest.raises(ValueError, match="source asset appears in more than one split"):
        audit_and_cache_avmit_features(
            archive_path,
            pair_dir,
            output,
            expected_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            expected_archive_size=archive_path.stat().st_size,
            expected_archive_md5=hashlib.md5(archive_path.read_bytes()).hexdigest(),
        )
    assert not output.exists()
