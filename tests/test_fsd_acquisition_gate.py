import hashlib
import json

import pytest

from scripts.fsd_acquisition_gate import complete_acquisition


def fixture(tmp_path):
    selection = [{"id": "123"}]
    data = json.dumps(selection).encode()
    (tmp_path / "selection.json").write_bytes(data)
    (tmp_path / "123.wav").write_bytes(b"fixture")
    report = {"status": "PASS", "selection": selection,
              "files": [{"id": "123", "sha256": hashlib.sha256(b"fixture").hexdigest()}]}
    (tmp_path / "result.json").write_text(json.dumps(report))
    return hashlib.sha256(data).hexdigest(), report


def test_valid_and_corrupted_media(tmp_path):
    digest, report = fixture(tmp_path)
    assert complete_acquisition(tmp_path, digest, 1) == report
    (tmp_path / "123.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="media hash"):
        complete_acquisition(tmp_path, digest, 1)


def test_changed_selection_and_partial_report(tmp_path):
    digest, report = fixture(tmp_path)
    with pytest.raises(ValueError, match="selection hash"):
        complete_acquisition(tmp_path, "wrong", 1)
    report["status"] = "RUNNING"
    (tmp_path / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="terminal"):
        complete_acquisition(tmp_path, digest, 1)


def test_missing_member_cannot_pass(tmp_path):
    digest, report = fixture(tmp_path)
    report["files"] = []
    (tmp_path / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity"):
        complete_acquisition(tmp_path, digest, 1)
