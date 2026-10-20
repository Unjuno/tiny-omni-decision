import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_ami_media_hash_overlap import audit_receipts


def _receipt_entry(meeting: str, filename: str, path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "meeting": meeting,
        "kind": "video",
        "file": filename,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_media(root: Path, meeting: str, filename: str, payload: bytes) -> Path:
    path = root / meeting / "video" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_audit_normalizes_receipt_schemas_and_reports_exact_cross_split_match(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    train_path = _write_media(media_root, "SA01", "a.avi", b"same frames")
    valid_path = _write_media(media_root, "SB01", "b.avi", b"same frames")
    other_path = _write_media(media_root, "SA02", "c.avi", b"different frames")
    first = tmp_path / "legacy.json"
    first.write_text(
        json.dumps({"files": [_receipt_entry("SA01", "a.avi", train_path)]}),
        encoding="utf-8",
    )
    second = tmp_path / "array.json"
    second.write_text(
        json.dumps(
            [
                _receipt_entry("SB01", "b.avi", valid_path),
                _receipt_entry("SA02", "c.avi", other_path),
            ]
        ),
        encoding="utf-8",
    )

    result = audit_receipts([first, second], media_root, {"SA01", "SA02"}, {"SB01"})

    assert result["record_count"] == 3
    assert result["train_record_count"] == 2
    assert result["validation_record_count"] == 1
    assert result["cross_split_exact_duplicate_group_count"] == 1


def test_audit_fails_closed_on_receipt_hash_mismatch(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    path = _write_media(media_root, "SA01", "a.avi", b"actual")
    receipt = tmp_path / "receipt.json"
    entry = _receipt_entry("SA01", "a.avi", path)
    entry["sha256"] = "0" * 64
    receipt.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_receipts([receipt], media_root, {"SA01"}, set())


def test_audit_rejects_split_overlap_before_reading_media(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="meeting sets overlap"):
        audit_receipts([], tmp_path, {"SA01"}, {"SA01"})
