"""Verify a task-specific CLEVRER AV expansion split using existing train media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

DATASET = "MIT-IBM/CLEVRER"
REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
EXPERIMENT_ID = "clevrer-synthetic-av-coverage-v2"
SEED = 17


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _ordered_id_hash(values: list[int]) -> str:
    return _sha256_bytes("\n".join(str(value) for value in values).encode("ascii"))


def _split_hash(values: list[int]) -> str:
    payload = "\n".join(str(value) for value in sorted(values)).encode("ascii")
    return _sha256_bytes(payload)


def _media_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe relative media path: {relative}")
    path = root.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"media path escapes media root: {relative}") from error
    return path


def preflight(
    *,
    expanded_train_path: Path,
    expanded_media_root: Path,
    previous_train_path: Path,
    previous_validation_path: Path,
    previous_media_receipt_path: Path,
    validation_scene_count: int = 200,
) -> dict[str, Any]:
    expanded_rows = _read_jsonl(expanded_train_path)
    previous_rows = _read_jsonl(previous_train_path) + _read_jsonl(previous_validation_path)
    previous_media_receipt_bytes = previous_media_receipt_path.read_bytes()
    previous_receipt = json.loads(previous_media_receipt_bytes)

    if validation_scene_count < 1:
        raise ValueError("validation_scene_count must be positive")
    if (
        previous_receipt.get("status") != "complete_train_only_media_fetch"
        or previous_receipt.get("dataset") != DATASET
        or previous_receipt.get("revision") != REVISION
        or previous_receipt.get("license") != "CC0-1.0"
        or previous_receipt.get("validation_test_or_sealed_data_opened") is not False
    ):
        raise ValueError("previous AV media receipt is not the pinned train-only receipt")

    by_scene: dict[int, dict[str, Any]] = {}
    for row in expanded_rows:
        if row.get("source") != f"{DATASET}@{REVISION}" or row.get("split") != "train":
            raise ValueError(
                "expanded candidate rows must come from the pinned official train split"
            )
        if not str(row.get("id", "")).startswith("clevrer-train-"):
            raise ValueError("expanded candidate row is not an official-train CLEVRER record")
        scene = row.get("scene_index")
        if not isinstance(scene, int) or isinstance(scene, bool) or scene < 0:
            raise ValueError("scene_index must be a non-negative integer")
        identity = (row.get("media_path"), row.get("media_sha256"), row.get("media_bytes"))
        prior = by_scene.setdefault(scene, {"identity": identity, "rows": 0})
        if prior["identity"] != identity:
            raise ValueError(f"inconsistent media identity across rows for scene {scene}")
        prior["rows"] += 1

    previous_scenes: set[int] = set()
    for row in previous_rows:
        if row.get("source") != f"{DATASET}@{REVISION}" or row.get("split") not in {
            "train",
            "validation",
        }:
            raise ValueError(
                "previous AV rows must be from the pinned CLEVRER official-train source"
            )
        scene = row.get("scene_index")
        if not isinstance(scene, int) or isinstance(scene, bool):
            raise ValueError("previous AV row has an invalid scene ID")
        previous_scenes.add(scene)
    previous_media = {
        int(entry["scene_index"]): str(entry["sha256"])
        for entry in previous_receipt.get("entries", [])
    }
    if not previous_media or not previous_scenes <= set(previous_media):
        raise ValueError("previous AV sample scenes are not covered by its media receipt")

    expanded_scenes = set(by_scene)
    scene_overlap = expanded_scenes & previous_scenes
    if scene_overlap:
        raise ValueError(
            f"expanded candidate overlaps previous AV scene IDs: {sorted(scene_overlap)[:5]}"
        )
    expanded_hashes: dict[str, int] = {}
    media_bytes_total = 0
    for scene, value in sorted(by_scene.items()):
        relative, expected_hash, expected_size = value["identity"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"invalid media SHA-256 for scene {scene}")
        if not isinstance(expected_size, int) or expected_size < 16:
            raise ValueError(f"invalid media byte count for scene {scene}")
        path = _media_path(expanded_media_root, str(relative))
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"missing or wrong-sized media file for scene {scene}: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"media SHA-256 mismatch for scene {scene}")
        if actual_hash in expanded_hashes:
            raise ValueError(
                "duplicate expanded media content for scenes "
                f"{expanded_hashes[actual_hash]} and {scene}"
            )
        if actual_hash in set(previous_media.values()):
            raise ValueError(f"expanded media content overlaps previous AV scene {scene}")
        expanded_hashes[actual_hash] = scene
        media_bytes_total += expected_size

    if len(expanded_scenes) < validation_scene_count + 1:
        raise ValueError("candidate pool is too small for disjoint train and validation groups")
    validation_scenes = sorted(
        expanded_scenes,
        key=lambda scene: hashlib.sha256(
            f"{SEED}|{EXPERIMENT_ID}|validation|{scene}".encode("ascii")
        ).hexdigest(),
    )[:validation_scene_count]
    validation_set = set(validation_scenes)
    candidate_train_scenes = sorted(expanded_scenes - validation_set)
    combined_train_scenes = sorted(previous_scenes | set(candidate_train_scenes))
    if set(combined_train_scenes) & validation_set:
        raise ValueError("task-specific train and development scene groups overlap")

    return {
        "status": "PASS_MEDIA_AND_TASK_SPLIT_PREFLIGHT_ONLY",
        "experiment_id": EXPERIMENT_ID,
        "source": f"{DATASET}@{REVISION}",
        "license": "CC0-1.0",
        "seed": SEED,
        "previous_av_rows_reused_as_training": True,
        "previous_av_validation_is_reclassified_as_training": True,
        "new_validation_is_task_specific_not_media_unseen_or_blind": True,
        "sealed_or_official_validation_test_opened": False,
        "expanded_candidate_input_rows": len(expanded_rows),
        "expanded_candidate_unique_scenes": len(expanded_scenes),
        "expanded_candidate_media_bytes_verified": media_bytes_total,
        "expanded_candidate_scene_sha256": _split_hash(sorted(expanded_scenes)),
        "expanded_candidate_order_sha256": _ordered_id_hash(sorted(expanded_scenes)),
        "expanded_candidate_previous_av_scene_overlap": 0,
        "expanded_candidate_previous_av_media_hash_overlap": 0,
        "previous_av_scenes_used_for_new_training": len(previous_scenes),
        "candidate_train_scenes": len(candidate_train_scenes),
        "candidate_train_scene_ids": candidate_train_scenes,
        "candidate_validation_scenes": len(validation_scenes),
        "candidate_validation_scene_ids": validation_scenes,
        "candidate_validation_scene_sha256": _split_hash(validation_scenes),
        "combined_task_training_scenes": len(combined_train_scenes),
        "combined_task_training_scene_sha256": _split_hash(combined_train_scenes),
        "previous_media_receipt_sha256": _sha256_bytes(previous_media_receipt_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-train", type=Path, required=True)
    parser.add_argument("--expanded-media-root", type=Path, required=True)
    parser.add_argument("--previous-train", type=Path, required=True)
    parser.add_argument("--previous-validation", type=Path, required=True)
    parser.add_argument("--previous-media-receipt", type=Path, required=True)
    parser.add_argument("--validation-scenes", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.output}")
    result = preflight(
        expanded_train_path=args.expanded_train,
        expanded_media_root=args.expanded_media_root,
        previous_train_path=args.previous_train,
        previous_validation_path=args.previous_validation,
        previous_media_receipt_path=args.previous_media_receipt,
        validation_scene_count=args.validation_scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if "_ids" not in key}, indent=2))


if __name__ == "__main__":
    main()
