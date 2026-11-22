import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_soundactions_split import build_manifest, verify_publisher_media_checksums


def test_soundactions_split_is_deterministic_and_disjoint():
    first = build_manifest()
    second = build_manifest()
    assert first == second
    splits = first["splits"]
    assert set(splits) == {"train", "validation", "sealed_audit"}
    assert sum(first["counts"].values()) == 365
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["sealed_audit"])
    assert set(splits["validation"]).isdisjoint(splits["sealed_audit"])
    all_ids = set(splits["train"]) | set(splits["validation"]) | set(splits["sealed_audit"])
    assert all_ids == set(range(1, 366))


def test_soundactions_exposed_preview_rows_are_train_only():
    manifest = build_manifest()
    assert set(range(1, 36)) <= set(manifest["splits"]["train"])
    assert not (set(range(1, 36)) & set(manifest["splits"]["validation"]))
    assert not (set(range(1, 36)) & set(manifest["splits"]["sealed_audit"]))
    assert manifest["known_exposure"]["no_other_per_record_label_values_used_for_assignment"]


def test_soundactions_manifest_hash_covers_split_payload():
    manifest = build_manifest()
    digest = manifest.pop("manifest_sha256")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == digest


def test_checked_in_soundactions_split_manifest_matches_frozen_assignments():
    path = (
        Path(__file__).resolve().parents[1]
        / "manifests/pretrained_reuse/365-sound-actions-split-v1-20261121.json"
    )
    checked = json.loads(path.read_text(encoding="utf-8"))
    digest = checked.pop("manifest_sha256")
    canonical = json.dumps(
        checked, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == digest
    assert checked["splits"] == build_manifest()["splits"]
    assert checked["sealed_audit_accessed"] is False
    assert checked["media_integrity"]["status"] == "PASS_EXACT_BYTES"


def test_publisher_checksums_reject_cross_split_exact_media_overlap(tmp_path: Path):
    same = "a" * 64
    rows = []
    for sample_id, title in ((1, "same recording"), (2, "duplicate bytes")):
        for modality, extension in (("JPG", "jpg"), ("MP4", "mp4"), ("WAV", "wav")):
            digest = same if modality == "WAV" else f"{sample_id:064x}"
            rows.append(f"{digest}  {modality}/SoundAction {sample_id:03d} - {title}.{extension}")
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    splits = {"train": [1], "validation": [2], "sealed_audit": []}
    with pytest.raises(ValueError, match="identical publisher media hashes"):
        verify_publisher_media_checksums(path, splits)


def test_publisher_checksum_audit_counts_all_modalities(tmp_path: Path):
    rows = []
    for sample_id, title in ((1, "first"), (2, "second")):
        for modality, extension in (("JPG", "jpg"), ("MP4", "mp4"), ("WAV", "wav")):
            digest = hashlib.sha256(f"{modality}|{sample_id}".encode()).hexdigest()
            rows.append(f"{digest}  {modality}/SoundAction {sample_id:03d} - {title}.{extension}")
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = verify_publisher_media_checksums(
        path, {"train": [1], "validation": [2], "sealed_audit": []}
    )
    assert result["status"] == "PASS_EXACT_BYTES"
    assert result["media_files_checked"] == 6
    assert result["cross_recording_byte_hash_collisions"] == 0
