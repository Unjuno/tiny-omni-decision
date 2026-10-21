"""Build scene-disjoint synthetic AV v2 samples from frozen official-train annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.create_clevrer_synthetic_av_protocol import (
    DATASET,
    REVISION,
    SAMPLES_PER_VIDEO_FRAME,
    SEQUENCE_FRAMES,
    eligible_collision_events,
    scene_time_shifts,
)

OLD_TRAIN_MANIFEST_SHA256 = "f2ac5c836ac387661367e9caee150dec58684aa93f7df4e150aa574efd294e03"
OLD_VALIDATION_MANIFEST_SHA256 = "b3436be649dc02ec825f488a710dac78d99f9dee36783b71d2ace74bb13a4b1a"
OLD_FETCH_REPORT_SHA256 = "a46e10865127f49be2048c1ba021118650def24b8a806154fb7750b4d6248b16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _new_event_rows(
    scene: int,
    split: str,
    media: dict[str, Any],
    annotation: dict[str, Any],
    shift: int,
) -> list[dict[str, Any]]:
    rows = []
    for event in eligible_collision_events(annotation):
        left, right = event["object_descriptors"]
        question = (
            "Did the sound happen before or after the collision between "
            f"the {left['color']} {left['material']} {left['shape']} and "
            f"the {right['color']} {right['material']} {right['shape']}?"
        )
        pair_id = f"{event['event_id']}-paired-av-timing"
        for relation, offset, target in (
            ("sound_before_collision", -16, 0),
            ("sound_after_collision", 16, 1),
        ):
            onset_frame = shift + event["frame_id"] + offset
            if not 0 <= onset_frame < SEQUENCE_FRAMES:
                raise ValueError(f"sound onset escapes frozen sequence for scene {scene}")
            sample_id = f"{pair_id}-{relation}"
            rows.append(
                {
                    "id": sample_id,
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "event_id": event["event_id"],
                    "source": f"{DATASET}@{REVISION}",
                    "split": split,
                    "scene_index": scene,
                    "scene_group_id": f"clevrer:train:{scene}",
                    "media_path": media["media_path"],
                    "media_sha256": media["media_sha256"],
                    "question": question,
                    "options": ["sound before collision", "sound after collision"],
                    "target": "sound before collision" if target == 0 else "sound after collision",
                    "target_index": target,
                    "relation": relation,
                    "collision_frame": event["frame_id"],
                    "object_ids": event["object_ids"],
                    "object_descriptors": event["object_descriptors"],
                    "scene_time_shift_frames": shift,
                    "video_frame_range": [shift, shift + 127],
                    "sound_onset_frame": onset_frame,
                    "sound_onset_sample": onset_frame * SAMPLES_PER_VIDEO_FRAME,
                }
            )
    return rows


def _validate_pairs(rows: list[dict[str, Any]], split: str) -> None:
    pairs: dict[str, list[dict[str, Any]]] = {}
    ids = set()
    for row in rows:
        if row["id"] in ids or row["split"] != split:
            raise ValueError(f"duplicate sample or split mismatch in {split}")
        ids.add(row["id"])
        pairs.setdefault(row["pair_id"], []).append(row)
    for pair_id, members in pairs.items():
        if (
            len(members) != 2
            or {row["target_index"] for row in members} != {0, 1}
            or len({row["scene_group_id"] for row in members}) != 1
            or len({row["media_sha256"] for row in members}) != 1
            or len({row["question"] for row in members}) != 1
        ):
            raise ValueError(f"incomplete paired event {pair_id}")


def materialize_v2(
    *,
    preflight_path: Path,
    annotation_dir: Path,
    annotation_receipt_path: Path,
    old_sample_dir: Path,
    expanded_train_manifest: Path,
    expanded_media_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite sample output: {output_dir}")
    split_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_receipt_path.read_text(encoding="utf-8"))
    if (
        split_record.get("status") != "PASS_MEDIA_AND_TASK_SPLIT_PREFLIGHT_ONLY"
        or annotations.get("status") != "PASS_PINNED_TRAIN_ANNOTATIONS_EXTRACTED"
        or annotations.get("source") != f"{DATASET}@{REVISION}"
        or annotations.get("official_split") != "train"
        or annotations.get("sealed_or_official_validation_test_opened") is not False
        or annotations.get("preflight_sha256") != sha256_file(preflight_path)
    ):
        raise ValueError("preflight/annotation receipts do not establish pinned train-only input")
    split_train = set(split_record["candidate_train_scene_ids"])
    split_validation = set(split_record["candidate_validation_scene_ids"])
    if split_train & split_validation:
        raise ValueError("scene split overlaps")
    entries = {Path(item["file"]).name: item for item in annotations["entries"]}
    if len(entries) != annotations["scene_count"]:
        raise ValueError("annotation receipt has missing or duplicate scene files")
    media_rows = _load_jsonl(expanded_train_manifest)
    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in media_rows:
        relative_media = PurePosixPath(str(row.get("media_path", "")))
        if (
            row.get("source") != f"{DATASET}@{REVISION}"
            or row.get("split") != "train"
            or not str(row.get("id", "")).startswith("clevrer-train-")
            or relative_media.is_absolute()
            or ".." in relative_media.parts
            or len(str(row.get("media_sha256", ""))) != 64
            or not isinstance(row.get("media_bytes"), int)
            or row["media_bytes"] < 16
        ):
            raise ValueError("expanded media manifest contains an invalid or non-train row")
        scene = int(row["scene_index"])
        prior = media_by_scene.get(scene)
        item = {
            "media_path": row["media_path"],
            "media_sha256": row["media_sha256"],
            "media_bytes": row["media_bytes"],
        }
        if prior is not None and prior != item:
            raise ValueError(f"expanded media identity differs within scene {scene}")
        media_by_scene[scene] = item
    if set(media_by_scene) != split_train | split_validation:
        raise ValueError("expanded official-train media scene IDs differ from frozen preflight")

    new_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    shifts = scene_time_shifts(split_train | split_validation, seed=17)
    new_scene_events: dict[int, int] = {}
    for scene in sorted(split_train | split_validation):
        split = "train" if scene in split_train else "validation"
        media = media_by_scene[scene]
        media_path = expanded_media_root / Path(media["media_path"])
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if media_path.stat().st_size != media["media_bytes"] or sha256_file(media_path) != media[
            "media_sha256"
        ]:
            raise ValueError(f"official-train media hash mismatch for scene {scene}")
        name = f"annotation_{scene:05d}.json"
        receipt_entry = entries.get(name)
        annotation_path = annotation_dir / name
        if not receipt_entry or not annotation_path.is_file():
            raise FileNotFoundError(f"receipt-linked annotation is missing: {name}")
        if annotation_path.stat().st_size != receipt_entry["bytes"] or sha256_file(
            annotation_path
        ) != receipt_entry["sha256"]:
            raise ValueError(f"annotation hash mismatch for scene {scene}")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("scene_index") != scene or annotation.get("video_filename") != (
            f"video_{scene:05d}.mp4"
        ):
            raise ValueError(f"annotation identity mismatch for scene {scene}")
        events = eligible_collision_events(annotation)
        new_scene_events[scene] = len(events)
        new_rows[split].extend(_new_event_rows(scene, split, media, annotation, shifts[scene]))

    old_train = _load_jsonl(old_sample_dir / "train.jsonl")
    old_validation = _load_jsonl(old_sample_dir / "validation.jsonl")
    old_report = json.loads((old_sample_dir / "fetch-report.json").read_text(encoding="utf-8"))
    old_rows = old_train + old_validation
    if (
        sha256_file(old_sample_dir / "train.jsonl") != OLD_TRAIN_MANIFEST_SHA256
        or sha256_file(old_sample_dir / "validation.jsonl") != OLD_VALIDATION_MANIFEST_SHA256
        or sha256_file(old_sample_dir / "fetch-report.json") != OLD_FETCH_REPORT_SHA256
        or len(old_rows) != 1986
        or len({int(row["scene_index"]) for row in old_rows}) != 401
        or len(old_train) != 1596
        or len(old_validation) != 390
        or old_report.get("validation_test_or_sealed_data_opened") is not False
    ):
        raise ValueError("old v1 corpus differs from its frozen 401-scene train-only record")
    candidate_scene_ids = split_train | split_validation
    old_scene_ids = {int(row["scene_index"]) for row in old_rows}
    old_media = {str(row["media_sha256"]) for row in old_rows}
    new_media = {str(media_by_scene[scene]["media_sha256"]) for scene in candidate_scene_ids}
    if old_scene_ids & candidate_scene_ids or old_media & new_media:
        raise ValueError("old synthetic AV and added official-train media identities overlap")
    for original in old_rows:
        row = dict(original)
        scene = int(row["scene_index"])
        row["split"] = "train"
        row["scene_group_id"] = f"clevrer:train:{scene}"
        row["id"] = str(row.get("id", row["sample_id"]))
        row["sample_id"] = row["id"]
        new_rows["train"].append(row)

    for split in ("train", "validation"):
        new_rows[split].sort(key=lambda row: row["id"])
        _validate_pairs(new_rows[split], split)
    train_scenes = {row["scene_group_id"] for row in new_rows["train"]}
    validation_scenes = {row["scene_group_id"] for row in new_rows["validation"]}
    train_media = {row["media_sha256"] for row in new_rows["train"]}
    validation_media = {row["media_sha256"] for row in new_rows["validation"]}
    if train_scenes & validation_scenes or train_media & validation_media:
        raise ValueError("v2 sample split has scene/media leakage")

    output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in new_rows[split]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    question_sets = {
        split: {" ".join(row["question"].casefold().split()) for row in rows}
        for split, rows in new_rows.items()
    }
    report = {
        "status": "complete_clevrer_synthetic_av_v2_sample_manifest",
        "dataset": f"{DATASET}@{REVISION}",
        "official_source_split": "train only",
        "seed": 17,
        "preflight_sha256": sha256_file(preflight_path),
        "annotation_receipt_sha256": sha256_file(annotation_receipt_path),
        "expanded_train_manifest_sha256": sha256_file(expanded_train_manifest),
        "old_v1_sample_manifest_sha256": {
            name: sha256_file(old_sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "old_v1_validation_reclassified_as_training": True,
        "train_sample_count": len(new_rows["train"]),
        "validation_sample_count": len(new_rows["validation"]),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "train_candidate_scene_ids": sorted(old_scene_ids | split_train),
        "train_sample_scene_ids": sorted({int(row["scene_index"]) for row in new_rows["train"]}),
        "validation_scene_ids": sorted(split_validation),
        "train_sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in new_rows["train"]).encode()
        ).hexdigest(),
        "validation_sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in new_rows["validation"]).encode()
        ).hexdigest(),
        "eligible_new_scene_count_by_split": {
            "train": sum(bool(new_scene_events[scene]) for scene in split_train),
            "validation": sum(bool(new_scene_events[scene]) for scene in split_validation),
        },
        "new_event_count_by_split": {
            "train": sum(new_scene_events[scene] for scene in split_train),
            "validation": sum(new_scene_events[scene] for scene in split_validation),
        },
        "question_template_overlap_count": len(
            question_sets["train"] & question_sets["validation"]
        ),
        "scene_overlap": 0,
        "media_hash_overlap": 0,
        "validation_is_task_specific_not_media_unseen_or_blind": True,
        "sealed_or_official_validation_test_opened": False,
    }
    (output_dir / "fetch-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--annotation-receipt", type=Path, required=True)
    parser.add_argument("--old-sample-dir", type=Path, required=True)
    parser.add_argument("--expanded-train-manifest", type=Path, required=True)
    parser.add_argument("--expanded-media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_v2(
        preflight_path=args.preflight.resolve(strict=True),
        annotation_dir=args.annotation_dir.resolve(strict=True),
        annotation_receipt_path=args.annotation_receipt.resolve(strict=True),
        old_sample_dir=args.old_sample_dir.resolve(strict=True),
        expanded_train_manifest=args.expanded_train_manifest.resolve(strict=True),
        expanded_media_root=args.expanded_media_root.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
