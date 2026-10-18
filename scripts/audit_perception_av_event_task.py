"""Audit audio-event to visible-object candidates in pinned Perception Test splits."""

import argparse
import collections
import json
import platform
import zipfile
from pathlib import Path

from scripts.audit_perception_av_correspondence import event_candidate, sha256_file

TRAIN_SOURCE_SHA256 = {
    "annotations": "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78",
    "cuts": "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4",
}
VALID_SOURCE_SHA256 = {
    "annotations": "eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7",
    "cuts": "3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7",
}
MEDIA_MANIFEST_SHA256 = {
    "train": "533c4cc0f41ca944a73306defd54c06299c30d242ec41ca2e43f3de5ea7f2eff",
    "valid": "957dcff3264fc5dd00ae63b13be9f5a3f5d81c81ed1cd5f8af31505a8be8fd3a",
}
MEDIA_PLAN_SHA256 = "0c6a07048c393d2fddd54fb3e72883055bb43634f9e3a8cac5d8570edcc39931"


def summarize_split(data, cuts, split):
    counts = collections.Counter()
    by_label = collections.Counter()
    option_counts = collections.Counter()
    candidate_videos = set()
    candidate_records = []
    for video_id, row in sorted(data.items()):
        metadata = row.get("metadata", {})
        if metadata.get("split") != split or metadata.get("video_id") != video_id:
            raise ValueError(f"{split} video identity/split mismatch")
        if video_id not in cuts or type(cuts[video_id]) is not int:
            raise ValueError(f"{split} pinned cut mapping required for every video")
        tracks = row.get("object_tracking", [])
        track_ids = [track.get("id") for track in tracks]
        if any(type(track_id) is not int or track_id < 0 for track_id in track_ids):
            raise ValueError("object track IDs must be nonnegative integers")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("duplicate object track ID")
        seen_sound_ids = set()
        for sound in row.get("sound_localisation", []):
            sound_id = sound.get("id")
            if type(sound_id) is not int or sound_id < 0 or sound_id in seen_sound_ids:
                raise ValueError("sound segment IDs must be unique nonnegative integers per video")
            seen_sound_ids.add(sound_id)
            decision = event_candidate(sound, tracks, cuts[video_id])
            counts[decision["status"]] += 1
            counts["sound_segments"] += 1
            if decision["reason"]:
                counts["held:" + decision["reason"]] += 1
            if decision["status"] != "candidate":
                continue
            candidate_videos.add(video_id)
            by_label[sound["label"]] += 1
            option_counts[len(decision["candidate_track_ids"])] += 1
            candidate_records.append(
                {
                    "video_id": video_id,
                    "sound_id": sound_id,
                    "sound_label": sound["label"],
                    "source_parent_ids": list(sound["parent_objects"]),
                    "frame_interval": list(sound["frame_ids"]),
                    **decision,
                }
            )
    if counts["candidate"] + counts["held"] != counts["sound_segments"]:
        raise ValueError("lost sound annotations")
    return {
        "split": split,
        "sound_segments": counts["sound_segments"],
        "candidate_sound_segments": counts["candidate"],
        "held_sound_segments": counts["held"],
        "held_reasons": {
            key.removeprefix("held:"): value
            for key, value in sorted(counts.items())
            if key.startswith("held:")
        },
        "candidate_video_ids": sorted(candidate_videos),
        "candidate_videos": len(candidate_videos),
        "eligible_by_sound_label": dict(sorted(by_label.items())),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "candidate_records": candidate_records,
    }


def inspect_media_manifest(path, expected_split):
    manifest_bytes_sha256 = sha256_file(path)
    if manifest_bytes_sha256 != MEDIA_MANIFEST_SHA256[expected_split]:
        raise ValueError(f"{expected_split} media manifest hash does not match pin")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_count = {"train": 256, "valid": 200}[expected_split]
    if (
        manifest.get("status") != "PASS_ACQUISITION_NOT_DECODE"
        or manifest.get("split") != expected_split
        or manifest.get("plan_sha256") != MEDIA_PLAN_SHA256
        or len(manifest.get("clips", [])) != expected_count
    ):
        raise ValueError(f"unexpected {expected_split} media manifest")
    rows = {}
    for clip in manifest["clips"]:
        video_id = clip.get("video_id")
        media = Path(clip.get("path", ""))
        if (
            not isinstance(video_id, str)
            or video_id in rows
            or clip.get("metadata", {}).get("split") != expected_split
            or clip.get("metadata", {}).get("video_id") != video_id
            or not media.is_file()
            or media.stat().st_size != clip.get("bytes")
        ):
            raise ValueError(f"invalid or missing {expected_split} media for {video_id}")
        rows[video_id] = {
            "path": str(media),
            "sha256": clip.get("sha256"),
            "bytes": media.stat().st_size,
        }
    return manifest_bytes_sha256, rows


def load_annotations(archive_path, cut_path, expected_split, expected_sha256):
    actual = {"annotations": sha256_file(archive_path), "cuts": sha256_file(cut_path)}
    if actual != expected_sha256:
        raise ValueError(f"{expected_split} annotation source hashes do not match pins")
    expected_member = f"all_{expected_split}.json"
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != [expected_member]:
            raise ValueError(f"unexpected {expected_split} archive members")
        data = json.loads(archive.read(expected_member))
    cuts = json.loads(cut_path.read_text(encoding="utf-8"))
    return data, cuts, actual


def media_coverage(summary, media_ids):
    candidate_ids = set(summary["candidate_video_ids"])
    covered = candidate_ids & media_ids
    records = [
        row for row in summary["candidate_records"] if row["video_id"] in covered
    ]
    return {
        "candidate_videos_in_selected_media": len(covered),
        "candidate_events_in_selected_media": len(records),
        "candidate_events_by_sound_label": dict(
            sorted(collections.Counter(row["sound_label"] for row in records).items())
        ),
        "candidate_option_count_distribution": dict(
            sorted(collections.Counter(len(row["candidate_track_ids"]) for row in records).items())
        ),
        "media_video_ids_with_candidate_events": sorted(covered),
        "candidate_event_records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--train-media-manifest", type=Path, required=True)
    parser.add_argument("--valid-media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required")

    train_data, train_cuts, train_hashes = load_annotations(
        args.train_source / "train_annotations.zip",
        args.train_source / "cut_frame_mapping_train.json",
        "train",
        TRAIN_SOURCE_SHA256,
    )
    valid_pins = json.loads((args.valid_source / "pins.json").read_text(encoding="utf-8"))
    valid_hashes_expected = VALID_SOURCE_SHA256
    if (
        valid_pins["annotations"]["sha256"] != VALID_SOURCE_SHA256["annotations"]
        or valid_pins["cuts"]["sha256"] != VALID_SOURCE_SHA256["cuts"]
    ):
        raise ValueError("valid-source receipt does not match committed pins")
    valid_data, valid_cuts, valid_hashes = load_annotations(
        args.valid_source / "valid_annotations.zip",
        args.valid_source / "cut_frame_mapping_valid.json",
        "valid",
        valid_hashes_expected,
    )
    train_summary = summarize_split(train_data, train_cuts, "train")
    valid_summary = summarize_split(valid_data, valid_cuts, "valid")
    if set(train_data) & set(valid_data):
        raise ValueError("train/valid video IDs overlap")

    train_manifest_sha256, train_media = inspect_media_manifest(
        args.train_media_manifest, "train"
    )
    valid_manifest_sha256, valid_media = inspect_media_manifest(
        args.valid_media_manifest, "valid"
    )
    if set(train_media) & set(valid_media):
        raise ValueError("train/valid media IDs overlap")
    train_coverage = media_coverage(train_summary, set(train_media))
    valid_coverage = media_coverage(valid_summary, set(valid_media))

    args.output.mkdir(parents=True, exist_ok=False)
    records_path = args.output / "candidate-events.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for split, coverage in (("train", train_coverage), ("valid", valid_coverage)):
            for record in coverage.pop("candidate_event_records"):
                stream.write(
                    json.dumps({"official_split": split, **record}, sort_keys=True) + "\n"
                )
    report = {
        "status": "PASS_EVENT_TASK_DATA_FEASIBILITY",
        "source_revision": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
        "train_source_sha256": train_hashes,
        "valid_source_sha256": valid_hashes,
        "media_manifest_sha256": {
            "train": train_manifest_sha256,
            "valid": valid_manifest_sha256,
        },
        "media_plan_sha256": MEDIA_PLAN_SHA256,
        "train": {
            key: value for key, value in train_summary.items() if key != "candidate_records"
        },
        "valid": {
            key: value for key, value in valid_summary.items() if key != "candidate_records"
        },
        "selected_media_coverage": {"train": train_coverage, "valid": valid_coverage},
        "candidate_records_sha256": sha256_file(records_path),
        "candidate_rule": (
            "one resolved non-Muted source parent, is_visible=1, before split-specific cut; "
            "choose label-blind annotated frame nearest sound interval midpoint among frames "
            "with at least two visible nonempty boxes; hold if source is absent at selected frame"
        ),
        "option_scope": (
            "supplied ground-truth object tracks; this is conditional AV source selection"
        ),
        "grouping": "all sound events from one video stay in its official split",
        "participant_disjointness": "UNKNOWN; public annotation records lack participant IDs",
        "valid_scope": (
            "official development validation; assets were previously used for a "
            "different recording-level AV task"
        ),
        "test_opened": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "model_loaded": False,
        "training": False,
        "python_version": platform.python_version(),
    }
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
