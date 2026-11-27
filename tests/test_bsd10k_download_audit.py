from __future__ import annotations

import hashlib
import json
import wave
import zlib

from scripts.audit_bsd10k_downloads import audit_downloaded_splits


def _write_wav(
    path,
    pcm: bytes,
    *,
    add_junk_chunk: bool = False,
    channels: int = 1,
    rate: int = 44_100,
    width: int = 2,
) -> dict:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(pcm)
    if add_junk_chunk:
        raw = path.read_bytes()
        raw = raw[:36] + b"JUNK\x04\x00\x00\x00test" + raw[36:]
        raw = raw[:4] + (len(raw) - 8).to_bytes(4, "little") + raw[8:]
        path.write_bytes(raw)
    return {
        "bytes": path.stat().st_size,
        "crc32": f"{zlib.crc32(path.read_bytes()):08x}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _run(tmp_path, *, same_pcm: bool) -> dict:
    root = tmp_path / "media"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir()
    train_pcm = bytes(range(128)) * 4
    val_pcm = train_pcm if same_pcm else bytes(reversed(range(128))) * 4
    train_hash = _write_wav(root / "train" / "1.wav", train_pcm)
    val_hash = _write_wav(root / "validation" / "2.wav", val_pcm, add_junk_chunk=same_pcm)
    manifest = {
        "schema": "tiny-omni-decision/bsd10k-group-split-v1",
        "source_record_id": "19868804",
        "examples": [
            {
                "sound_id": "1",
                "class": "fx-o",
                "license": "CC0-1.0",
                "uploader_group_sha256": "a" * 64,
                "split": "train",
            },
            {
                "sound_id": "2",
                "class": "fx-o",
                "license": "CC-BY-4.0",
                "uploader_group_sha256": "b" * 64,
                "split": "validation",
            },
        ],
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_paths = []
    for split, sound_id, file_hashes, license_name in (
        ("train", "1", train_hash, "CC0-1.0"),
        ("validation", "2", val_hash, "CC-BY-4.0"),
    ):
        result_path = tmp_path / f"{split}.json"
        result_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "source_record_id": "19868804",
                    "downloaded_count": 1,
                    "downloaded_files": [
                        {
                            "sound_id": sound_id,
                            "license": license_name,
                            "class": "fx-o",
                            **file_hashes,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(result_path)
    return audit_downloaded_splits(manifest_path, result_paths, root)


def test_download_audit_checks_media_hashes_and_split_groups(tmp_path) -> None:
    report = _run(tmp_path, same_pcm=False)
    assert report["cross_split_sound_id_overlap_count"] == 0
    assert report["cross_split_uploader_group_overlap_count"] == 0
    assert report["cross_split_file_sha256_overlap_count"] == 0
    assert report["cross_split_normalized_pcm_overlap_count"] == 0
    assert report["data_quality_status"] == "PASS_NO_OVERLAP"


def test_download_audit_rejects_identical_pcm_across_splits(tmp_path) -> None:
    report = _run(tmp_path, same_pcm=True)
    assert report["cross_split_file_sha256_overlap_count"] == 0
    assert report["cross_split_normalized_pcm_overlap_count"] == 1
    assert report["data_quality_status"] == "FAIL_CROSS_SPLIT_OVERLAP"


def test_download_audit_accepts_valid_24bit_stereo_pcm_source(tmp_path) -> None:
    root = tmp_path / "media"
    (root / "train").mkdir(parents=True)
    file_hash = _write_wav(root / "train" / "1.wav", bytes(24), channels=2, rate=48_000, width=3)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "tiny-omni-decision/bsd10k-group-split-v1",
                "source_record_id": "19868804",
                "examples": [
                    {
                        "sound_id": "1",
                        "class": "fx-o",
                        "license": "CC-BY-3.0",
                        "uploader_group_sha256": "a" * 64,
                        "split": "train",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "train.json"
    result_path.write_text(
        json.dumps(
            {
                "split": "train",
                "source_record_id": "19868804",
                "downloaded_count": 1,
                "downloaded_files": [
                    {"sound_id": "1", "license": "CC-BY-3.0", "class": "fx-o", **file_hash}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_downloaded_splits(
        manifest_path, [result_path], root, extra_allowed_licenses={"CC-BY-3.0"}
    )

    assert report["completed_splits"]["train"]["pcm_format_counts"] == {"(48000, 2, 3, 4)": 1}
