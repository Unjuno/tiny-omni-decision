from copy import deepcopy

import pytest

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.verify_starss23_acquisition import validate_acquisition


def fixture(tmp_path):
    row = dict(
        clip="a", split="train", room="r", upstream_split="train", source_revision="doi:source"
    )
    pair = dict(row, files={})
    for kind in ("audio", "video"):
        path = tmp_path / kind
        path.write_bytes(kind.encode())
        pair["files"][kind] = dict(
            path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path)
        )
    return [row], dict(status="PASS", pairs=[pair])


def test_complete_paired_assets(tmp_path):
    rows, report = fixture(tmp_path)
    assert validate_acquisition(report, rows)["unique_media_assets"] == 2


@pytest.mark.parametrize("status", ["INDEXING", "FETCHING", "FAIL", "PREFLIGHT_PASS"])
def test_partial_run_rejected(tmp_path, status):
    rows, report = fixture(tmp_path)
    report["status"] = status
    with pytest.raises(ValueError, match="incomplete"):
        validate_acquisition(report, rows)


def test_split_tampering_and_missing_modality_rejected(tmp_path):
    rows, report = fixture(tmp_path)
    changed = deepcopy(report)
    changed["pairs"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_acquisition(changed, rows)
    del report["pairs"][0]["files"]["audio"]
    with pytest.raises(ValueError, match="paired"):
        validate_acquisition(report, rows)


def test_media_tampering_rejected(tmp_path):
    rows, report = fixture(tmp_path)
    (tmp_path / "audio").write_bytes(b"xxxxx")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_acquisition(report, rows)


def test_duplicate_media_rejected_across_recordings(tmp_path):
    rows, report = fixture(tmp_path)
    second = dict(rows[0], clip="b", split="validation", room="other")
    rows.append(second)
    report["pairs"].append(dict(second, files=deepcopy(report["pairs"][0]["files"])))
    with pytest.raises(ValueError, match="duplicate underlying"):
        validate_acquisition(report, rows)
