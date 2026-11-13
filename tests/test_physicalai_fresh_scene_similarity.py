from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.audit_physicalai_fresh_scene_similarity import (
    render_neighbor_contact_sheet,
    render_representative_contact_sheet,
    representative_sample_ids,
    selected_frame_indices,
    validate_inputs,
)


def _fixture() -> tuple[dict, dict]:
    train = list(range(4000))
    validation = list(range(4000, 5000))
    assignments = []
    files = []
    for episode, split in [(value, "train") for value in train] + [
        (value, "validation") for value in validation
    ]:
        path = f"videos/{episode}.mp4"
        digest = hashlib.sha256(path.encode()).hexdigest()
        assignments.append(
            {
                "episode_index": episode,
                "split": split,
                "video_path": path,
                "video_bytes": 123,
                "video_sha256": digest,
            }
        )
        files.append(
            {
                "episode": episode,
                "split": split,
                "kind": "video",
                "path": path,
                "size": 123,
                "sha256": digest,
                "local_path": f"{episode}.mp4",
            }
        )
        files.append({"episode": episode, "split": split, "kind": "parquet"})
    manifest = {
        "status": "REVIEW_NOT_ADMITTED",
        "reserved_development_episode_indices": list(range(5000, 6000)),
        "episode_assignments": assignments,
    }
    receipt = {
        "dataset_id": "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm",
        "revision": "73c00c18e7ca108e91846294b20a1d7cd5de9340",
        "subset": "panda-stack-platforms",
        "split_manifest_sha256": "manifest-hash",
        "selection": {
            "train_episode_indices": train,
            "validation_episode_indices": validation,
        },
        "reserved_holdout_payload_downloaded": False,
        "files": files,
    }
    return receipt, manifest


def test_selected_frame_indices_are_fixed_temporal_quantiles() -> None:
    assert selected_frame_indices(101) == (20, 50, 80)


@pytest.mark.parametrize("frame_count", [0, 1, 2])
def test_selected_frame_indices_reject_too_short_video(frame_count: int) -> None:
    with pytest.raises(ValueError):
        selected_frame_indices(frame_count)


def test_neighbor_contact_sheet_uses_only_matching_train_validation_rows() -> None:
    thumbnails = np.zeros((2, 3, 32, 32, 3), dtype="uint8")
    image = render_neighbor_contact_sheet(
        thumbnails,
        np.asarray([11, 22]),
        np.asarray(["validation", "train"]),
        [
            {
                "validation_episode": 11,
                "train_episode": 22,
                "cosine_similarity": 0.9,
            }
        ],
    )
    assert image.size == (6 * 72, 20 + 72)


def test_representative_sample_is_deterministic_and_split_specific() -> None:
    ids = np.asarray([4, 8, 12, 16, 20, 24])
    splits = np.asarray(["train", "train", "train", "validation", "validation", "validation"])
    first = representative_sample_ids(ids, splits, count=2)
    assert first == representative_sample_ids(ids, splits, count=2)
    assert len(first["train"]) == len(first["validation"]) == 2
    sheet = render_representative_contact_sheet(
        np.zeros((6, 3, 32, 32, 3), dtype="uint8"), ids, splits, first
    )
    assert sheet.size == (3 * 64, 4 * 82)


def test_validate_inputs_accepts_only_exact_train_validation_video_receipt() -> None:
    receipt, manifest = _fixture()
    rows = validate_inputs(
        receipt, manifest, receipt_sha="receipt-hash", manifest_sha="manifest-hash"
    )
    assert len(rows) == 5000
    assert sum(row["split"] == "train" for row in rows) == 4000
    assert sum(row["split"] == "validation" for row in rows) == 1000


def test_validate_inputs_rejects_reserved_or_mismatched_video() -> None:
    receipt, manifest = _fixture()
    receipt["selection"]["validation_episode_indices"][0] = 5000
    with pytest.raises(ValueError, match="overlap"):
        validate_inputs(receipt, manifest, receipt_sha="receipt-hash", manifest_sha="manifest-hash")

    receipt, manifest = _fixture()
    receipt["files"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest"):
        validate_inputs(receipt, manifest, receipt_sha="receipt-hash", manifest_sha="manifest-hash")


def test_validate_inputs_rejects_reserved_payload_access() -> None:
    receipt, manifest = _fixture()
    receipt["reserved_holdout_payload_downloaded"] = True
    with pytest.raises(ValueError, match="reserved"):
        validate_inputs(receipt, manifest, receipt_sha="receipt-hash", manifest_sha="manifest-hash")


def test_validate_inputs_enforces_optional_exact_file_pins() -> None:
    receipt, manifest = _fixture()
    with pytest.raises(ValueError, match="pinned acquisition receipt"):
        validate_inputs(
            receipt,
            manifest,
            receipt_sha="receipt-hash",
            manifest_sha="manifest-hash",
            expected_receipt_sha="different-receipt-hash",
        )
