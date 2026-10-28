import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from scripts.acquire_grid_video_archives import file_hashes
from scripts.audit_grid_media_duplicates import (
    archive_member_hashes,
    digest_stream,
    find_duplicates,
    run_audit,
)


def test_digest_stream_hashes_complete_payload():
    assert digest_stream(io.BytesIO(b"payload")) == hashlib.sha256(b"payload").hexdigest()


def test_archive_member_hashes_reads_audio_and_video_payloads(tmp_path):
    audio_path = tmp_path / "audio.tar"
    with tarfile.open(audio_path, "w") as archive:
        payload = b"wav bytes"
        info = tarfile.TarInfo("audio/s2/bbaa1a.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    video_path = tmp_path / "video.zip"
    with zipfile.ZipFile(video_path, "w") as archive:
        archive.writestr("s2/bbaa1a.mpg", b"mpg bytes")

    assert archive_member_hashes(audio_path, ".wav") == {
        "bbaa1a": hashlib.sha256(b"wav bytes").hexdigest()
    }
    assert archive_member_hashes(video_path, ".mpg") == {
        "bbaa1a": hashlib.sha256(b"mpg bytes").hexdigest()
    }


def test_archive_member_hashes_reject_duplicate_media_identity(tmp_path):
    path = tmp_path / "audio.tar"
    with tarfile.open(path, "w") as archive:
        for name in ("a/bbaa1a.wav", "b/bbaa1a.wav"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="duplicate .wav payload"):
        archive_member_hashes(path, ".wav")


def test_find_duplicates_flags_cross_split_but_not_cross_modality_match():
    shared = hashlib.sha256(b"same audio").hexdigest()
    records = [
        {"modality": "audio", "sha256": shared, "split": "train", "code": "a"},
        {"modality": "audio", "sha256": shared, "split": "validation", "code": "b"},
        {"modality": "video", "sha256": shared, "split": "validation", "code": "c"},
    ]

    result = find_duplicates(records)

    assert len(result["cross_split_exact_payload_duplicates"]) == 1
    assert result["cross_split_exact_payload_duplicates"][0]["modality"] == "audio"
    assert result["within_split_exact_payload_duplicates"] == []


def test_run_audit_validates_all_inputs_and_scans_each_frozen_split(tmp_path):
    speakers = [f"s{index}" for index in range(1, 33)]
    train_speakers, validation_speakers = speakers[:24], speakers[24:]
    source_receipts, bundle_speakers, archive_receipts = {}, {}, []
    rows = {"train": [], "validation": []}
    for index, speaker in enumerate(speakers, start=1):
        split = "train" if index <= 24 else "validation"
        code = f"c{index}"
        audio_data = f"audio-{speaker}".encode()
        video_data = f"video-{speaker}".encode()
        audio_path = tmp_path / f"{speaker}-audio.tar"
        with tarfile.open(audio_path, "w") as archive:
            info = tarfile.TarInfo(f"audio/{code}.wav")
            info.size = len(audio_data)
            archive.addfile(info, io.BytesIO(audio_data))
        video_path = tmp_path / f"{speaker}.zip"
        with zipfile.ZipFile(video_path, "w") as archive:
            archive.writestr(f"video/{code}.mpg", video_data)
        audio_sha = file_hashes(audio_path)[1]
        video_md5, video_sha = file_hashes(video_path)
        bundle_speakers[speaker] = {
            "audio_tar": str(audio_path),
            "audio_sha256": audio_sha,
        }
        archive_receipts.append(
            {
                "speaker": speaker,
                "path": str(video_path),
                "sha256": video_sha,
                "publisher_md5": video_md5,
            }
        )
        source_receipts[speaker] = {
            "video_archive_bytes": video_path.stat().st_size,
            "video_publisher_md5": f"md5:{video_md5}",
            "codes_in_audio_alignment_video": 1,
        }
        rows[split].append({"speaker": speaker, "code": code, "targets": {}})

    bundle_index_path = tmp_path / "bundle-index.json"
    bundle_index_path.write_text(
        json.dumps({"speakers": bundle_speakers}), encoding="utf-8"
    )
    train_path, validation_path = tmp_path / "train.json", tmp_path / "validation.json"
    train_path.write_text(json.dumps(rows["train"]), encoding="utf-8")
    validation_path.write_text(json.dumps(rows["validation"]), encoding="utf-8")
    summary = {
        "frozen_speaker_assignment": {
            "train": train_speakers,
            "validation": validation_speakers,
        },
        "source_receipts": source_receipts,
        "bundle_index_sha256": file_hashes(bundle_index_path)[1],
        "output_sha256": {
            "train": file_hashes(train_path)[1],
            "validation": file_hashes(validation_path)[1],
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_sha = file_hashes(summary_path)[1]
    acquired = {
        "status": "PASS_ARCHIVES_VERIFIED",
        "source_summary_sha256": summary_sha,
        "archives": archive_receipts,
    }
    archive_result_path = tmp_path / "archive-result.json"
    archive_result_path.write_text(json.dumps(acquired), encoding="utf-8")

    report = run_audit(
        summary_path=summary_path,
        expected_summary_sha256=summary_sha,
        bundle_index_path=bundle_index_path,
        archive_result_path=archive_result_path,
        train_path=train_path,
        validation_path=validation_path,
        output_path=tmp_path / "report.json",
    )

    assert report["status"] == "PASS_NO_CROSS_SPLIT_EXACT_MEDIA_DUPLICATES"
    assert report["asset_payload_count"] == 64
    assert report["cross_split_exact_payload_duplicate_groups"] == 0
