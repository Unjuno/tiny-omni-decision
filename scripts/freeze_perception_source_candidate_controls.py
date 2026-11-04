"""Freeze label-blind AV source-candidate intervention mappings for a fixed split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate_records(records, split_name):
    ids = [record["record_id"] for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError(f"{split_name} records must have unique IDs and be nonempty")
    for record in records:
        if record.get("status") != "candidate" or record.get("split") != "train":
            raise ValueError("control inputs must come from candidate rows in official train")
        candidate_ids = [candidate["candidate_id"] for candidate in record["request"]["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate ID in {record['record_id']}")
        targets = record["supervision"]["target_candidate_ids"]
        if not set(targets).issubset(candidate_ids):
            raise ValueError(f"target is outside candidate set in {record['record_id']}")
        if not targets and not record["supervision"]["none_visible"]:
            raise ValueError(f"candidate row has no positive/none target: {record['record_id']}")


def build_split_controls(records, clips, *, seed, split_name, source_sha256):
    _validate_records(records, split_name)
    by_group = defaultdict(list)
    for record in records:
        video_id = record["group_id"]
        if video_id not in clips:
            raise ValueError(f"media receipt missing for {split_name} group {video_id}")
        by_group[video_id].append(record)
    if len(by_group) < 2:
        raise ValueError(f"{split_name} needs at least two video groups for audio shuffle")

    ordered_groups = sorted(by_group, key=lambda group: _rank(seed, f"{split_name}:group:{group}"))
    donor_group = {
        group: ordered_groups[(index + 1) % len(ordered_groups)]
        for index, group in enumerate(ordered_groups)
    }
    intervals = {}
    for record in records:
        interval = record["request"]["audio_interval_us"]
        if len(interval) != 2 or interval[1] <= interval[0]:
            raise ValueError(f"invalid audio interval in {record['record_id']}")
        intervals[record["record_id"]] = interval

    serialized_records = []
    for group in sorted(by_group):
        own_records = sorted(by_group[group], key=lambda row: row["record_id"])
        donor_records = sorted(
            by_group[donor_group[group]], key=lambda row: _rank(seed, f"donor:{row['record_id']}")
        )
        clip = clips[group]
        metadata = clip["metadata"]
        audio_duration_us = round(metadata["audio_samples"] / metadata["audio_sample_rate"] * 1e6)
        video_duration_us = round(metadata["num_frames"] / metadata["frame_rate"] * 1e6)
        cut_frame = clip.get("cut_frame", -1)
        cut_duration_us = (
            video_duration_us
            if cut_frame == -1
            else round(cut_frame / metadata["frame_rate"] * 1e6)
        )
        for index, record in enumerate(own_records):
            interval = intervals[record["record_id"]]
            offset_rank = int(_rank(seed, f"offset:{record['record_id']}")[:2], 16)
            preferred_sign = 1 if offset_rank % 2 else -1
            offset_us = None
            for sign in (preferred_sign, -preferred_sign):
                shifted = [interval[0] + sign * 1_000_000, interval[1] + sign * 1_000_000]
                if (
                    shifted[0] >= 0
                    and shifted[1] <= audio_duration_us
                    and shifted[1] <= video_duration_us
                    and shifted[1] <= cut_duration_us
                ):
                    offset_us = sign * 1_000_000
                    break

            candidates = record["request"]["candidates"]
            original_order = [candidate["candidate_id"] for candidate in candidates]
            permuted = sorted(
                original_order,
                key=lambda candidate_id: _rank(
                    seed, f"{split_name}:{record['record_id']}:{candidate_id}"
                ),
            )
            if len(permuted) > 1 and permuted == original_order:
                permuted = permuted[1:] + permuted[:1]
            static_frames = {
                candidate["candidate_id"]: candidate["frame_boxes"][0]["frame_id"]
                for candidate in candidates
            }
            reversed_frames = {
                candidate["candidate_id"]: [
                    box["frame_id"] for box in reversed(candidate["frame_boxes"])
                ]
                for candidate in candidates
            }
            donor = donor_records[index % len(donor_records)]
            serialized_records.append(
                {
                    "record_id": record["record_id"],
                    "video_id": group,
                    "audio_interval_us": interval,
                    "audio_shuffle_donor_record_id": donor["record_id"],
                    "audio_shuffle_donor_video_id": donor_group[group],
                    "audio_shuffle_donor_interval_us": intervals[donor["record_id"]],
                    "offset_audio_interval_us": (
                        [interval[0] + offset_us, interval[1] + offset_us]
                        if offset_us is not None
                        else None
                    ),
                    "candidate_order": permuted,
                    "static_frame_id_by_candidate": static_frames,
                    "reversed_frame_ids_by_candidate": reversed_frames,
                }
            )

    if any(row["video_id"] == row["audio_shuffle_donor_video_id"] for row in serialized_records):
        raise AssertionError("audio shuffle must use a different video group")
    return {
        "record_count": len(serialized_records),
        "video_group_count": len(by_group),
        "audio_shuffle_derangement": True,
        "offset_eligible_count": sum(
            row["offset_audio_interval_us"] is not None for row in serialized_records
        ),
        "candidate_order_changed_count": sum(
            row["candidate_order"]
            != [candidate["candidate_id"] for candidate in record["request"]["candidates"]]
            for row, record in zip(
                serialized_records,
                sorted(records, key=lambda item: (item["group_id"], item["record_id"])),
                strict=True,
            )
        ),
        "frame_reversal_eligible_count": sum(
            any(len(candidate["frame_boxes"]) > 1 for candidate in record["request"]["candidates"])
            for record in records
        ),
        "records": serialized_records,
        "source_jsonl_sha256": source_sha256,
    }


def freeze_controls(train_path, validation_path, acquisition_manifest_path, *, seed=17):
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    acquisition_manifest_path = Path(acquisition_manifest_path)
    acquisition = json.loads(acquisition_manifest_path.read_text(encoding="utf-8"))
    clips = {}
    for clip in acquisition["clips"]:
        metadata = clip["metadata"]
        video_id = metadata["video_id"]
        if metadata.get("split") != "train" or video_id in clips:
            raise ValueError("control media receipt must be unique official-train clips")
        if clip.get("bytes") <= 0 or not clip.get("sha256"):
            raise ValueError(f"media receipt lacks integrity fields for {video_id}")
        clips[video_id] = clip

    train = _read_jsonl(train_path)
    validation = _read_jsonl(validation_path)
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("train and development video groups overlap")
    return {
        "format": "perception_test_source_candidate_controls_v1",
        "seed": seed,
        "source_split": "official train only; non-blind grouped development",
        "controls": [
            "unmodified joint audio-video candidate input",
            "audio ablation",
            "video ablation",
            "within-split cross-video audio shuffle",
            "same-video one-second audio interval offset when media bounds permit",
            "candidate-order permutation",
            "static-first-frame candidate crops",
            "reversed candidate-frame order",
        ],
        "selection_uses_labels_or_metrics": False,
        "model_loaded": False,
        "metrics_computed": False,
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "train": build_split_controls(
            train,
            clips,
            seed=seed,
            split_name="train",
            source_sha256=sha256_file(train_path),
        ),
        "development": build_split_controls(
            validation,
            clips,
            seed=seed,
            split_name="development",
            source_sha256=sha256_file(validation_path),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--acquisition-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior control manifest: {output}")
    manifest = freeze_controls(
        args.train,
        args.development,
        args.acquisition_manifest,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in manifest.items()
        if key in {"format", "seed", "source_split", "controls", "selection_uses_labels_or_metrics"}
    }
    summary["train"] = {key: value for key, value in manifest["train"].items() if key != "records"}
    summary["development"] = {
        key: value for key, value in manifest["development"].items() if key != "records"
    }
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
