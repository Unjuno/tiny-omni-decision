from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_clevrer_scene_exposure import (
    _cached_scene_ids,
    _excluded,
    _json_exposures,
    candidate_scene_ids,
    exposed_scene_ids,
    iter_train_scene_ids,
    sha256_ids,
    write_receipt,
)


def test_streams_train_scene_indices_and_checks_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "train-questions.json"
    source.write_text(json.dumps([{"scene_index": 0}, {"scene_index": 9}]), encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert list(iter_train_scene_ids(source, expected_sha256=expected)) == [0, 9]
    with pytest.raises(ValueError, match="SHA-256"):
        list(iter_train_scene_ids(source, expected_sha256="0" * 64))


def test_exposure_extracts_source_and_parent_scene_ids() -> None:
    rows = [
        {"source": "clevrer:video_00003", "id": "q-1"},
        {"scene_index": 7, "id": "CLEVRER-00007-question-2"},
        {"source_id": "clevrer/video_00011", "id": "q-3"},
    ]

    assert exposed_scene_ids(rows) == {3, 7, 11}


def test_exposure_ignores_incidental_clevrer_text_without_source_identity() -> None:
    rows = [
        {"question": "CLEVRER asks a question", "scene_index": 5, "id": "unrelated-5"},
        {"source": "CLEVRER", "scene_index": 6, "id": "train-6"},
    ]

    assert exposed_scene_ids(rows) == {6}


def test_json_exposure_inherits_dataset_identity_and_reads_split_manifests(tmp_path: Path) -> None:
    report = tmp_path / "clevrer-report.json"
    report.write_text(
        json.dumps({"dataset_id": "MIT-IBM/CLEVRER", "rows": [{"scene_index": 4}]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "clevrer-heldout-video-excluded-scenes-20261011.json"
    manifest.write_text(json.dumps({"train": [8], "validation": [9]}), encoding="utf-8")

    exposed, row_count = _json_exposures([report, manifest])

    assert exposed == {4, 8, 9}
    assert row_count == 5


def test_json_exposure_accepts_newline_json_stored_with_json_suffix(tmp_path: Path) -> None:
    report = tmp_path / "clevrer-export.json"
    report.write_text(
        '{"source":"CLEVRER","scene_index":12}\n'
        '{"source":"CLEVRER","scene_index":13}\n',
        encoding="utf-8",
    )

    exposed, _ = _json_exposures([report])

    assert exposed == {12, 13}


def test_json_exposure_skips_malformed_non_dataset_report(tmp_path: Path) -> None:
    report = tmp_path / "status.json"
    report.write_text('{"status": true}\\n', encoding="utf-8")

    exposed, row_count = _json_exposures([report])

    assert exposed == set()
    assert row_count == 0


def test_candidates_exclude_any_previously_exposed_or_cached_scenes() -> None:
    assert candidate_scene_ids(range(8), exposed={2, 4}, cached_media={3, 7}) == [0, 1, 5, 6]


def test_cached_media_scan_uses_filenames_without_opening_media(tmp_path: Path) -> None:
    (tmp_path / "video_00042.mp4").write_bytes(b"not a video; filename is the audit signal")
    (tmp_path / "video_12000.mp4").write_bytes(b"outside official train range")

    assert _cached_scene_ids([tmp_path]) == {42, 12_000}


def test_audit_excludes_sealed_and_test_named_metadata_files(tmp_path: Path) -> None:
    assert _excluded(tmp_path / "sealed-audit.jsonl")
    assert _excluded(tmp_path / "test-rows.json")
    assert not _excluded(tmp_path / "validation-rows.json")


def test_scene_hash_is_order_independent_and_receipt_never_overwrites(tmp_path: Path) -> None:
    assert sha256_ids([4, 1, 2]) == sha256_ids([1, 2, 4])
    target = tmp_path / "receipt.json"
    write_receipt(target, {"candidate_scene_ids": [1, 3]})
    with pytest.raises(FileExistsError):
        write_receipt(target, {"candidate_scene_ids": [2]})
    assert json.loads(target.read_text(encoding="utf-8"))["candidate_scene_ids"] == [1, 3]
