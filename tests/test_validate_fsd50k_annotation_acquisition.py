import hashlib
import json
import zlib

import pytest

from scripts.build_fsd50k_annotation_decision_manifest import (
    DATASET_REVISION,
    canonical_sha256,
    summarize_split,
)
from scripts.validate_fsd50k_annotation_acquisition import validate_acquisition


def fixture(root):
    preflight = root / "preflight"
    media = root / "media"
    preflight.mkdir()
    media.mkdir()
    splits = {
        "train": [_row("100", "train-uploader", "train")],
        "calibration": [_row("200", "cal-uploader", "calibration")],
        "validation": [_row("300", "val-uploader", "validation")],
    }
    summaries = {name: summarize_split(rows) for name, rows in splits.items()}
    source_sha = {"FSD50K.metadata.zip": "a" * 64}
    manifest_hash = canonical_sha256({"splits": summaries, "source_sha256": source_sha})
    preflight_report = {
        "status": "PREFLIGHT_PASS",
        "dataset_revision": DATASET_REVISION,
        "source_sha256": source_sha,
        "splits": summaries,
        "manifest_sha256": manifest_hash,
    }
    (preflight / "preflight.json").write_text(json.dumps(preflight_report), encoding="utf-8")
    for split, rows in splits.items():
        (preflight / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    receipts = []
    payload_by_split = {"train": b"wav-train", "calibration": b"wav-cal", "validation": b"wav-val"}
    for split, rows in splits.items():
        for row in rows:
            payload = payload_by_split[split]
            path = media / split / f"{row['sample_id']}.wav"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            receipts.append(
                {
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "group_id": row["group_id"],
                    "license": row["license"],
                    "split": split,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "zip_crc32": zlib.crc32(payload),
                }
            )
    receipt_bytes = "".join(json.dumps(row) + "\n" for row in receipts).encode()
    (media / "files.jsonl").write_bytes(receipt_bytes)
    (media / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "dataset_revision": DATASET_REVISION,
                "preflight_manifest_sha256": manifest_hash,
                "official_eval_labels_read": False,
                "official_eval_media_acquired": False,
                "expected_files": len(receipts),
                "downloaded_files": len(receipts),
                "files_jsonl_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return preflight, media


def _row(sample_id, uploader, split):
    return {
        "sample_id": sample_id,
        "source_id": f"freesound:{sample_id}",
        "group_id": uploader,
        "split": split,
        "license": "http://creativecommons.org/licenses/by/3.0/",
        "targets": {"Animal": False, "Human_voice": True, "Vehicle": False, "Music": False},
    }


def test_complete_acquisition_hashes_every_file_and_split(tmp_path):
    preflight, media = fixture(tmp_path)

    result = validate_acquisition(preflight, media)

    assert result["status"] == "PASS"
    assert result["verified_files"] == 3
    assert result["verified_files_by_split"] == {"train": 1, "calibration": 1, "validation": 1}
    assert result["cross_split_exact_media_overlap"] == 0


def test_corrupt_wav_fails_integrity_check(tmp_path):
    preflight, media = fixture(tmp_path)
    (media / "train" / "100.wav").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_acquisition(preflight, media)


def test_identical_audio_across_splits_fails(tmp_path):
    preflight, media = fixture(tmp_path)
    shared = b"same exact media"
    for split, sample_id in (("train", "100"), ("validation", "300")):
        (media / split / f"{sample_id}.wav").write_bytes(shared)
    receipt_path = media / "files.jsonl"
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    for row in receipts:
        if row["sample_id"] in {"100", "300"}:
            row.update(bytes=len(shared), sha256=hashlib.sha256(shared).hexdigest(),
                       zip_crc32=zlib.crc32(shared))
    payload = "".join(json.dumps(row) + "\n" for row in receipts).encode()
    receipt_path.write_bytes(payload)
    result_path = media / "result.json"
    result = json.loads(result_path.read_text())
    result["files_jsonl_sha256"] = hashlib.sha256(payload).hexdigest()
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="overlap across FSD50K splits"):
        validate_acquisition(preflight, media)


def test_eval_access_or_unlisted_media_fails(tmp_path):
    preflight, media = fixture(tmp_path)
    result_path = media / "result.json"
    result = json.loads(result_path.read_text())
    result["official_eval_labels_read"] = True
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="eval must remain unopened"):
        validate_acquisition(preflight, media)

    result["official_eval_labels_read"] = False
    result_path.write_text(json.dumps(result))
    (media / "validation" / "999.wav").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected or missing WAV"):
        validate_acquisition(preflight, media)
