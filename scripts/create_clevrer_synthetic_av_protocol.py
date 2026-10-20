"""Freeze a train-only synthetic CLEVRER AV timing diagnostic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET = "MIT-IBM/CLEVRER"
REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
EXPOSURE_RECEIPT_SHA256 = "0fc99660e3ea05095856696540ace7bbc5dc9899a4e4b612893a4edb69e65f38"
ANNOTATION_RECEIPT_SHA256 = "26c178bbfcbc06f38828d47592e7cce7a545c0372f4f9b7f676e7322a6e41485"
SCENE_IDS_SHA256 = "33bb4408f20ec95a64ed657da5e860588dccfd5378b95c52101ab0768baba4e8"
EXPECTED_CANDIDATE_SCENE_COUNT = 405
QUESTION_SOURCE_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
FRAME_RATE_NUMERATOR = 128
FRAME_RATE_DENOMINATOR = 5
AUDIO_SAMPLE_RATE = 16_000
SAMPLES_PER_VIDEO_FRAME = 625
COLLISION_OFFSET_FRAMES = 16
MIN_COLLISION_FRAME = COLLISION_OFFSET_FRAMES
MAX_COLLISION_FRAME = 127 - COLLISION_OFFSET_FRAMES
MAX_GLOBAL_SHIFT_FRAMES = 255
SEQUENCE_FRAMES = 384


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_ids(values: Iterable[int]) -> str:
    payload = ",".join(str(value) for value in sorted(set(values))).encode("ascii")
    return sha256_bytes(payload)


def sha256_sample_ids(values: Iterable[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return sha256_bytes(payload)


def scene_group_split(
    scene_ids: Iterable[int], seed: int = 17, validation_fraction: float = 0.2
) -> dict[str, list[int]]:
    values = list(scene_ids)
    ids = sorted(set(values))
    if len(ids) != len(values):
        raise ValueError("scene IDs must be unique")
    if not ids or not 0 < validation_fraction < 1:
        raise ValueError("scene IDs and a validation fraction in (0, 1) are required")
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = math.ceil(len(ids) * validation_fraction)
    validation = set(shuffled[:validation_count])
    return {
        "train": sorted(set(ids) - validation),
        "validation": sorted(validation),
    }


def scene_time_shifts(
    scene_ids: Iterable[int], seed: int = 17, max_shift_frames: int = MAX_GLOBAL_SHIFT_FRAMES
) -> dict[int, int]:
    if max_shift_frames < 0:
        raise ValueError("max_shift_frames must be non-negative")
    result = {}
    for scene in sorted(set(scene_ids)):
        digest = hashlib.sha256(f"{seed}|{scene}|shift".encode("ascii")).hexdigest()
        result[scene] = int(digest[:16], 16) % (max_shift_frames + 1)
    return result


def _object_signature(obj: dict[str, Any]) -> tuple[str, str, str] | None:
    values = tuple(
        str(obj.get(key, "")).strip().casefold() for key in ("color", "material", "shape")
    )
    return values if all(values) else None


def eligible_collision_events(
    annotation: dict[str, Any],
    min_frame: int = MIN_COLLISION_FRAME,
    max_frame: int = MAX_COLLISION_FRAME,
) -> list[dict[str, Any]]:
    scene = annotation.get("scene_index")
    if not isinstance(scene, int) or isinstance(scene, bool) or scene < 0:
        raise ValueError("annotation scene_index must be a non-negative integer")
    objects = annotation.get("object_property")
    collisions = annotation.get("collision")
    if not isinstance(objects, list) or not isinstance(collisions, list):
        raise ValueError("annotation requires object_property and collision arrays")

    by_id: dict[int, tuple[str, str, str]] = {}
    signatures: Counter[tuple[str, str, str]] = Counter()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id")
        signature = _object_signature(obj)
        if isinstance(object_id, int) and not isinstance(object_id, bool) and signature:
            by_id[object_id] = signature
            signatures[signature] += 1

    pair_counts: Counter[tuple[int, int]] = Counter()
    valid_collisions: list[tuple[int, int, int, tuple[int, int]]] = []
    for index, collision in enumerate(collisions):
        if not isinstance(collision, dict):
            continue
        object_ids = collision.get("object_ids")
        frame = collision.get("frame_id")
        if (
            not isinstance(object_ids, list)
            or len(object_ids) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in object_ids)
            or object_ids[0] == object_ids[1]
            or not isinstance(frame, int)
            or isinstance(frame, bool)
        ):
            continue
        pair = tuple(sorted(object_ids))
        pair_counts[pair] += 1
        if min_frame <= frame <= max_frame:
            valid_collisions.append((index, frame, pair[0], pair))

    events = []
    for index, frame, _first_object_id, pair in valid_collisions:
        if pair_counts[pair] != 1:
            continue
        if any(object_id not in by_id for object_id in pair):
            continue
        pair_signatures = [by_id[object_id] for object_id in pair]
        if any(signatures[signature] != 1 for signature in pair_signatures):
            continue
        descriptors = [
            {"color": signature[0], "material": signature[1], "shape": signature[2]}
            for signature in sorted(pair_signatures)
        ]
        events.append(
            {
                "event_id": f"clevrer-train-{scene:05d}-collision-{index:03d}",
                "scene_index": scene,
                "collision_index": index,
                "frame_id": frame,
                "object_ids": list(pair),
                "object_descriptors": descriptors,
            }
        )
    return events


def _validate_receipts(exposure_path: Path, annotation_receipt_path: Path) -> tuple[dict, dict]:
    exposure_bytes = exposure_path.read_bytes()
    annotation_bytes = annotation_receipt_path.read_bytes()
    if sha256_bytes(exposure_bytes) != EXPOSURE_RECEIPT_SHA256:
        raise ValueError("exposure receipt SHA-256 is not the audited attempt-04 receipt")
    if sha256_bytes(annotation_bytes) != ANNOTATION_RECEIPT_SHA256:
        raise ValueError("annotation receipt SHA-256 is not the audited attempt-01 receipt")
    exposure = json.loads(exposure_bytes)
    annotation_receipt = json.loads(annotation_bytes)
    if exposure.get("candidate_scene_ids_sha256") != SCENE_IDS_SHA256:
        raise ValueError("candidate scene ID hash differs from the exposure audit")
    candidate_ids = exposure.get("candidate_scene_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != EXPECTED_CANDIDATE_SCENE_COUNT
        or sha256_ids(candidate_ids) != SCENE_IDS_SHA256
    ):
        raise ValueError("candidate scene list is malformed or does not match the frozen hash")
    if annotation_receipt.get("source_dataset") != DATASET:
        raise ValueError("annotation source dataset changed")
    if annotation_receipt.get("source_revision") != REVISION:
        raise ValueError("annotation source revision changed")
    if annotation_receipt.get("license") != "CC0-1.0" or annotation_receipt.get("split") != "train":
        raise ValueError("annotation license/split is not the admitted train-only source")
    if annotation_receipt.get("validation_test_and_sealed_data_opened") is not False:
        raise ValueError("source receipt does not certify validation/test/sealed isolation")
    if annotation_receipt.get("video_bytes_opened") is not False:
        raise ValueError("annotation receipt indicates video bytes were opened")
    if annotation_receipt.get("exposure_receipt_sha256") != EXPOSURE_RECEIPT_SHA256:
        raise ValueError("annotation receipt is not linked to the audited exposure receipt")
    entries = annotation_receipt.get("entries")
    if not isinstance(entries, list) or {entry.get("scene_index") for entry in entries} != set(
        candidate_ids
    ):
        raise ValueError("annotation receipt scene set differs from the screened candidates")
    return exposure, annotation_receipt


def build_protocol(
    annotation_dir: Path,
    exposure_receipt_path: Path,
    annotation_receipt_path: Path,
    seed: int = 17,
) -> dict[str, Any]:
    if seed != 17:
        raise ValueError("the frozen synthetic AV protocol uses immutable seed 17")
    exposure, annotation_receipt = _validate_receipts(
        exposure_receipt_path, annotation_receipt_path
    )
    entry_by_scene = {entry["scene_index"]: entry for entry in annotation_receipt["entries"]}
    scene_ids = sorted(entry_by_scene)
    splits = scene_group_split(scene_ids, seed=seed)
    shifts = scene_time_shifts(scene_ids, seed=seed)
    events_by_scene: dict[int, list[dict[str, Any]]] = {}
    for scene in scene_ids:
        entry = entry_by_scene[scene]
        path = annotation_dir / Path(entry["archive_member"]).name
        if not path.is_file():
            raise FileNotFoundError(f"frozen annotation is missing: {path}")
        payload_bytes = path.read_bytes()
        if len(payload_bytes) != entry["bytes"] or sha256_bytes(payload_bytes) != entry["sha256"]:
            raise ValueError(f"frozen annotation hash mismatch for scene {scene}")
        annotation = json.loads(payload_bytes)
        if annotation.get("scene_index") != scene:
            raise ValueError(f"annotation scene identity mismatch for scene {scene}")
        if annotation.get("video_filename") != f"video_{scene:05d}.mp4":
            raise ValueError(f"annotation video identity mismatch for scene {scene}")
        events_by_scene[scene] = eligible_collision_events(annotation)

    samples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, split_scenes in splits.items():
        for scene in split_scenes:
            shift = shifts[scene]
            for event in events_by_scene[scene]:
                for relation, offset, target in (
                    ("sound_before_collision", -COLLISION_OFFSET_FRAMES, 0),
                    ("sound_after_collision", COLLISION_OFFSET_FRAMES, 1),
                ):
                    onset_frame = shift + event["frame_id"] + offset
                    if not 0 <= onset_frame < SEQUENCE_FRAMES:
                        raise ValueError("synthetic sound onset escapes the frozen sequence")
                    pair_id = f"{event['event_id']}-paired-av-timing"
                    left, right = event["object_descriptors"]
                    question = (
                        "Did the sound happen before or after the collision between "
                        f"the {left['color']} {left['material']} {left['shape']} and "
                        f"the {right['color']} {right['material']} {right['shape']}?"
                    )
                    samples[split].append(
                        {
                            "sample_id": f"{pair_id}-{relation}",
                            "pair_id": pair_id,
                            "event_id": event["event_id"],
                            "scene_index": scene,
                            "video_filename": f"video_{scene:05d}.mp4",
                            "question": question,
                            "options": ["sound before collision", "sound after collision"],
                            "target": target,
                            "relation": relation,
                            "collision_frame": event["frame_id"],
                            "object_ids": event["object_ids"],
                            "object_descriptors": event["object_descriptors"],
                            "scene_time_shift_frames": shift,
                            "video_start_frame": shift,
                            "video_end_frame": shift + 127,
                            "sound_onset_frame": onset_frame,
                            "sound_onset_sample": onset_frame * SAMPLES_PER_VIDEO_FRAME,
                            "split": split,
                        }
                    )
    for split in samples:
        samples[split].sort(key=lambda row: row["sample_id"])
    all_scene_sets = {name: set(values) for name, values in splits.items()}
    if all_scene_sets["train"] & all_scene_sets["validation"]:
        raise ValueError("scene-group split contains overlap")
    for split, rows in samples.items():
        pair_targets: dict[str, set[int]] = {}
        for row in rows:
            pair_targets.setdefault(row["pair_id"], set()).add(row["target"])
        if any(targets != {0, 1} for targets in pair_targets.values()):
            raise ValueError(f"{split} contains an incomplete paired target")

    train_threshold = _fit_audio_onset_threshold(samples["train"])
    train_metrics = _threshold_metrics(samples["train"], train_threshold)
    validation_metrics = _threshold_metrics(samples["validation"], train_threshold)
    events_count = {
        split: sum(len(events_by_scene[scene]) for scene in scenes)
        for split, scenes in splits.items()
    }
    return {
        "protocol_id": "clevrer-synthetic-av-timing-v1",
        "status": "frozen_protocol_pre_media",
        "created_from": {
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "license": "CC0-1.0",
            "source_split": "train only",
            "question_source_sha256": QUESTION_SOURCE_SHA256,
            "exposure_receipt_sha256": sha256_bytes(exposure_receipt_path.read_bytes()),
            "annotation_receipt_sha256": sha256_bytes(annotation_receipt_path.read_bytes()),
            "candidate_scene_ids_sha256": exposure["candidate_scene_ids_sha256"],
            "candidate_scene_count": len(scene_ids),
        },
        "isolation": {
            "validation_test_or_sealed_data_opened": False,
            "video_bytes_opened_before_protocol_freeze": False,
            "scene_split_grouping": "underlying CLEVRER video scene_index",
            "scene_overlap": 0,
            "normalized_question_overlap_claim": (
                "not applicable: generated query, no source questions"
            ),
            "candidate_certification_limit": (
                "screened as absent from scanned metadata and cached filenames; normalized media "
                "content identity was not independently hashed"
            ),
        },
        "split": {
            "seed": seed,
            "validation_fraction": 0.2,
            "scene_ids": {name: values for name, values in splits.items()},
            "scene_ids_sha256": {name: sha256_ids(values) for name, values in splits.items()},
            "event_counts": events_count,
            "sample_counts": {name: len(rows) for name, rows in samples.items()},
            "paired_event_counts": {
                name: len({row["pair_id"] for row in rows}) for name, rows in samples.items()
            },
            "sample_id_order_sha256": {
                name: sha256_sample_ids(row["sample_id"] for row in rows)
                for name, rows in samples.items()
            },
        },
        "task": {
            "description": (
                "Determine whether a synthetic sound onset precedes or follows a specified "
                "CLEVRER collision. Each collision has paired before/after variants; all paired "
                "variants stay in the same scene split."
            ),
            "options": ["sound before collision", "sound after collision"],
            "target_mapping": {"sound_before_collision": 0, "sound_after_collision": 1},
            "collision_offset_frames": COLLISION_OFFSET_FRAMES,
            "eligible_collision_frame_range_inclusive": [MIN_COLLISION_FRAME, MAX_COLLISION_FRAME],
            "repeated_object_pair_events_excluded": True,
            "query_source": "generated from uniquely identifying CLEVRER object attributes",
            "text_only_control": (
                "paired samples have identical question and options, opposite target"
            ),
            "video_only_control": "paired samples have identical video, opposite target",
            "audio_only_control": (
                "audio absolute-onset threshold baseline fit on train only; evaluate held-out "
                "scene split; later run a learned audio-only control"
            ),
        },
        "temporal_encoding": {
            "source_video_frames": 128,
            "source_frame_rate": f"{FRAME_RATE_NUMERATOR}/{FRAME_RATE_DENOMINATOR} fps",
            "sequence_frames": SEQUENCE_FRAMES,
            "per_scene_global_shift_range_inclusive": [0, MAX_GLOBAL_SHIFT_FRAMES],
            "global_shift_rule": "SHA256(f'{seed}|{scene_index}|shift') first 64 bits modulo 256",
            "video_placement": (
                "source frames occupy [shift, shift+127]; repeat boundary frames elsewhere"
            ),
            "audio_sample_rate_hz": AUDIO_SAMPLE_RATE,
            "samples_per_video_frame": SAMPLES_PER_VIDEO_FRAME,
            "audio_length_samples": SEQUENCE_FRAMES * SAMPLES_PER_VIDEO_FRAME,
            "duration_seconds": SEQUENCE_FRAMES * FRAME_RATE_DENOMINATOR / FRAME_RATE_NUMERATOR,
            "synthetic_waveform": {
                "type": "fixed Hann-windowed sine burst",
                "frequency_hz": 880,
                "duration_ms": 80,
                "peak_amplitude": 0.3,
                "same_waveform_for_both_pair_members": True,
                "only_pair_difference": "onset at collision frame minus/plus 16 frames",
            },
        },
        "preflight_shortcut_control": {
            "model": "single threshold on absolute sound onset frame; threshold chosen on train",
            "chosen_threshold_frame": train_threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "warning": "diagnostic shortcut only; not an encoder or model quality result",
        },
        "samples": samples,
        "limitations": [
            "CLEVRER has no natural audio; generated sound tests temporal binding only",
            "the generated question template repeats; this is not language generalization",
            "metadata/filename screening does not prove normalized media-content non-overlap",
            "no validation, test, sealed audit, model, or feature cache was opened or evaluated",
        ],
    }


def _fit_audio_onset_threshold(rows: list[dict[str, Any]]) -> int:
    onset_values = sorted({row["sound_onset_frame"] for row in rows})
    candidates = [onset_values[0] - 1]
    candidates.extend(
        (left + right) // 2 + 1 for left, right in zip(onset_values, onset_values[1:], strict=False)
    )
    candidates.append(onset_values[-1] + 1)
    # Earlier onset predicts class 0; ties are resolved by the smallest threshold.
    return max(candidates, key=lambda value: (_threshold_metrics(rows, value)["accuracy"], -value))


def _threshold_metrics(rows: list[dict[str, Any]], threshold: int) -> dict[str, float | int]:
    correct = sum(
        int((row["sound_onset_frame"] < threshold) == (row["target"] == 0)) for row in rows
    )
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
    }


def write_protocol(path: Path, protocol: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--exposure-receipt", type=Path, required=True)
    parser.add_argument("--annotation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    protocol = build_protocol(
        args.annotation_dir,
        args.exposure_receipt,
        args.annotation_receipt,
        seed=args.seed,
    )
    write_protocol(args.output, protocol)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "output": str(args.output.resolve()),
        "scenes": protocol["split"]["scene_ids"],
        "event_counts": protocol["split"]["event_counts"],
        "sample_counts": protocol["split"]["sample_counts"],
        "audio_only_threshold_baseline": protocol["preflight_shortcut_control"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
