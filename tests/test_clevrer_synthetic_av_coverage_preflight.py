from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.preflight_clevrer_synthetic_av_coverage_v2 import preflight

DATASET = "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(scene: int, media_path: str, digest: str, split: str) -> dict:
    return {
        "id": f"clevrer-train-{scene:05d}-q000",
        "source": DATASET,
        "split": split,
        "scene_index": scene,
        "media_path": media_path,
        "media_sha256": digest,
        "media_bytes": 32,
    }


def _fixture(tmp_path: Path, *, validation_scenes: int = 2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    expanded_root = tmp_path / "expanded"
    previous_root = tmp_path / "previous"
    expanded_root.mkdir()
    previous_root.mkdir()
    expanded_rows = []
    for scene in range(10, 16):
        relative = f"media/train/video_{scene:05d}.mp4"
        media = expanded_root / relative
        media.parent.mkdir(parents=True, exist_ok=True)
        content = f"video-{scene}".encode().ljust(32, b"x")
        media.write_bytes(content)
        expanded_rows.append(_row(scene, relative, hashlib.sha256(content).hexdigest(), "train"))
    previous_media = []
    previous_rows = []
    for scene in (1, 2):
        content = f"old-{scene}".encode().ljust(32, b"x")
        digest = hashlib.sha256(content).hexdigest()
        previous_media.append({"scene_index": scene, "sha256": digest})
        previous_rows.append(_row(scene, f"media/video_{scene:05d}.mp4", digest, "train"))
    _jsonl(tmp_path / "expanded-train.jsonl", expanded_rows)
    _jsonl(tmp_path / "previous-train.jsonl", previous_rows[:1])
    validation_row = _row(
        2,
        "media/video_00002.mp4",
        previous_media[1]["sha256"],
        "validation",
    )
    _jsonl(tmp_path / "previous-validation.jsonl", [validation_row])
    receipt = {
        "status": "complete_train_only_media_fetch",
        "dataset": "MIT-IBM/CLEVRER",
        "revision": "98b842082ba4f7c18b6b9e3f39145871782a65ef",
        "license": "CC0-1.0",
        "validation_test_or_sealed_data_opened": False,
        "entries": previous_media,
    }
    (tmp_path / "previous-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return expanded_root, tmp_path, validation_scenes


def _run(tmp_path: Path, validation_scenes: int = 2) -> dict:
    expanded_root, paths, validation_scenes = _fixture(
        tmp_path, validation_scenes=validation_scenes
    )
    return preflight(
        expanded_train_path=paths / "expanded-train.jsonl",
        expanded_media_root=expanded_root,
        previous_train_path=paths / "previous-train.jsonl",
        previous_validation_path=paths / "previous-validation.jsonl",
        previous_media_receipt_path=paths / "previous-receipt.json",
        validation_scene_count=validation_scenes,
    )


def test_split_is_deterministic_disjoint_and_reclassifies_prior_validation(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")
    assert first["candidate_validation_scene_ids"] == second["candidate_validation_scene_ids"]
    assert len(first["candidate_validation_scene_ids"]) == 2
    assert set(first["candidate_validation_scene_ids"]).isdisjoint({1, 2})
    assert first["candidate_train_scenes"] == 4
    assert first["previous_av_scenes_used_for_new_training"] == 2
    assert first["combined_task_training_scenes"] == 6
    assert first["previous_av_validation_is_reclassified_as_training"] is True
    assert first["new_validation_is_task_specific_not_media_unseen_or_blind"] is True


def test_rejects_new_validation_scene_count_that_leaves_no_train_group(tmp_path: Path) -> None:
    expanded_root, paths, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="too small"):
        preflight(
            expanded_train_path=paths / "expanded-train.jsonl",
            expanded_media_root=expanded_root,
            previous_train_path=paths / "previous-train.jsonl",
            previous_validation_path=paths / "previous-validation.jsonl",
            previous_media_receipt_path=paths / "previous-receipt.json",
            validation_scene_count=6,
        )


def test_rejects_media_hash_mismatch(tmp_path: Path) -> None:
    expanded_root, paths, _ = _fixture(tmp_path)
    media = expanded_root / "media/train/video_00010.mp4"
    media.write_bytes(b"tampered".ljust(32, b"x"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preflight(
            expanded_train_path=paths / "expanded-train.jsonl",
            expanded_media_root=expanded_root,
            previous_train_path=paths / "previous-train.jsonl",
            previous_validation_path=paths / "previous-validation.jsonl",
            previous_media_receipt_path=paths / "previous-receipt.json",
            validation_scene_count=2,
        )


def test_rejects_cross_split_scene_or_media_overlap(tmp_path: Path) -> None:
    expanded_root, paths, _ = _fixture(tmp_path)
    row = json.loads((paths / "expanded-train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["scene_index"] = 1
    row["id"] = "clevrer-train-00001-q001"
    (paths / "expanded-train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps previous AV scene"):
        preflight(
            expanded_train_path=paths / "expanded-train.jsonl",
            expanded_media_root=expanded_root,
            previous_train_path=paths / "previous-train.jsonl",
            previous_validation_path=paths / "previous-validation.jsonl",
            previous_media_receipt_path=paths / "previous-receipt.json",
            validation_scene_count=1,
        )
