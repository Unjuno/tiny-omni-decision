"""Plan fresh video-level media acquisition for AV event-source selection."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.audit_mmfit import RemoteZip
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import load_annotations, summarize_split

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
FEASIBILITY_SHA256 = "1477f3d6bd12546103272bc8e45f3cfb0916b9e9c9c11d97629328cfda2f641d"
ANNOTATION_SHA256 = {
    "train_annotations.zip": "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78",
    "cut_frame_mapping_train.json": (
        "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4"
    ),
    "valid_annotations.zip": "eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7",
    "cut_frame_mapping_valid.json": (
        "3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7"
    ),
}
MEDIA_COUNT = 400
MAX_CLIP_BYTES = 256_000_000
MAX_SPLIT_BYTES = 8_000_000_000
MEDIA_MANIFESTS = {
    "perception-correspondence-train-media-01": (
        "533c4cc0f41ca944a73306defd54c06299c30d242ec41ca2e43f3de5ea7f2eff"
    ),
    "perception-correspondence-valid-media-01": (
        "957dcff3264fc5dd00ae63b13be9f5a3f5d81c81ed1cd5f8af31505a8be8fd3a"
    ),
    "perception-media-probe-01": (
        "a3f8df093a27bc01de2d733bfe77b4aac8a2f66776be2546f5ec111fa0fdf25f"
    ),
    "perception-media-probe-02": (
        "22e9b47df36c1847259c565393999051427ac0edb1621b6fc2d0ddf56e4bb0a5"
    ),
    "perception-occlusion-train-media-01": (
        "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3"
    ),
    "perception-occlusion-valid-media-01": (
        "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546"
    ),
}
TRAIN_ARCHIVE_PIN = {
    "url": "https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip?generation=1685446268139925",
    "bytes": 28404087974,
    "etag": '"-CJXz5tD4nP8CEAE="',
    "generation": "1685446268139925",
}
VALID_ARCHIVE_PIN = {
    "url": "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip?generation=1685479088600941",
    "bytes": 75375143328,
    "etag": '"-CO3u6PLynf8CEAE="',
    "generation": "1685479088600941",
}


def stable_order(video_ids, split, purpose):
    return sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(
            f"17:perception-av-event-v1:{split}:{purpose}:{video_id}".encode()
        ).digest(),
    )


def select_fresh_ids(candidate_ids, existing_ids, split, count=MEDIA_COUNT):
    if split not in {"train", "valid"} or count < 4 or count % 4:
        raise ValueError("split must be train/valid and media count a multiple of four >=4")
    available = set(candidate_ids) - set(existing_ids)
    selected = stable_order(available, split, "acquisition")[:count]
    if len(selected) != count:
        raise ValueError(f"need {count} fresh {split} candidate videos; found {len(selected)}")
    return selected


def read_prior_media_ids(root=ROOT):
    ids = set()
    pins = {}
    for directory, expected_hash in MEDIA_MANIFESTS.items():
        manifest_path = root / directory / "manifest.json"
        if sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"previous media manifest changed: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clips = manifest.get("clips")
        if not isinstance(clips, list):
            raise ValueError(f"previous media inventory malformed: {manifest_path}")
        ids.update(clip["video_id"] for clip in clips)
        pins[directory] = expected_hash
    return ids, pins


def archive_rows(pin, selected_ids, annotations, cuts, split):
    by_stem = {}
    with zipfile.ZipFile(RemoteZip(pin["url"], pin["bytes"], pin["etag"])) as archive:
        for info in archive.infolist():
            if info.filename.lower().endswith(".mp4"):
                by_stem.setdefault(Path(info.filename).stem, []).append(info)
        rows = []
        for video_id in selected_ids:
            matches = by_stem.get(video_id, [])
            if len(matches) != 1:
                raise ValueError(f"expected exactly one source member for {video_id}")
            metadata = annotations[video_id]["metadata"]
            if metadata.get("split") != split or metadata.get("video_id") != video_id:
                raise ValueError(f"annotation split or identity mismatch for {video_id}")
            info = matches[0]
            if not 0 < info.file_size <= MAX_CLIP_BYTES:
                raise ValueError(f"per-file media limit exceeded for {video_id}; no replacement")
            rows.append(
                {
                    "video_id": video_id,
                    "member": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": info.CRC,
                    "metadata": metadata,
                    "cut_frame": cuts[video_id],
                }
            )
    size = sum(row["bytes"] for row in rows)
    if len(rows) != len(selected_ids) or size > MAX_SPLIT_BYTES:
        raise ValueError(f"invalid {split} selection size {size}; no replacement")
    return rows


def candidate_counts(summary):
    counts = {video_id: 0 for video_id in summary["candidate_video_ids"]}
    for record in summary["candidate_records"]:
        counts[record["video_id"]] += 1
    return counts


def create_plan(root=ROOT, media_count=MEDIA_COUNT):
    feasibility_dir = root / "perception-av-event-task-feasibility-01"
    feasibility_path = feasibility_dir / "result.json"
    if sha256_file(feasibility_path) != FEASIBILITY_SHA256:
        raise ValueError("event-task feasibility receipt changed")
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("status") != "PASS_EVENT_TASK_DATA_FEASIBILITY":
        raise ValueError("event-task feasibility audit did not pass")
    prior_ids, prior_manifest_pins = read_prior_media_ids(root)

    train_source = root / "perception-test-source-audit-01"
    valid_source = root / "perception-correspondence-valid-audit-01"
    train_data, train_cuts, _ = load_annotations(
        train_source / "train_annotations.zip",
        train_source / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    valid_data, valid_cuts, _ = load_annotations(
        valid_source / "valid_annotations.zip",
        valid_source / "cut_frame_mapping_valid.json",
        "valid",
        {
            "annotations": ANNOTATION_SHA256["valid_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_valid.json"],
        },
    )
    train_summary = summarize_split(train_data, train_cuts, "train")
    valid_summary = summarize_split(valid_data, valid_cuts, "valid")
    if (
        train_summary["candidate_video_ids"] != feasibility["train"]["candidate_video_ids"]
        or valid_summary["candidate_video_ids"] != feasibility["valid"]["candidate_video_ids"]
    ):
        raise ValueError("recomputed eligible video IDs differ from pinned feasibility audit")

    train_ids = select_fresh_ids(
        train_summary["candidate_video_ids"], prior_ids, "train", media_count
    )
    valid_ids = select_fresh_ids(
        valid_summary["candidate_video_ids"], prior_ids, "valid", media_count
    )
    if set(train_ids) & set(valid_ids):
        raise ValueError("train/valid selected video overlap")

    train_counts = candidate_counts(train_summary)
    valid_counts = candidate_counts(valid_summary)
    train_group_order = stable_order(train_ids, "train", "model-split")
    calibration_ids = set(train_group_order[-media_count // 4 :])
    train_rows = archive_rows(
        TRAIN_ARCHIVE_PIN, train_ids, train_data, train_cuts, "train"
    )
    valid_rows = archive_rows(
        VALID_ARCHIVE_PIN, valid_ids, valid_data, valid_cuts, "valid"
    )
    for row in train_rows:
        row["model_split"] = "calibration" if row["video_id"] in calibration_ids else "train"
        row["candidate_events"] = train_counts[row["video_id"]]
    for row in valid_rows:
        row["model_split"] = "development_validation"
        row["candidate_events"] = valid_counts[row["video_id"]]
    return {
        "status": "PASS_FRESH_EVENT_MEDIA_PLAN",
        "task_id": "perception-av-event-source-selection-v1",
        "seed": 17,
        "source_revision": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
        "feasibility_result_sha256": FEASIBILITY_SHA256,
        "annotation_source_sha256": ANNOTATION_SHA256,
        "previous_media_manifest_sha256": prior_manifest_pins,
        "previous_unique_video_ids_excluded": len(prior_ids),
        "selection_rule": (
            "eligible video IDs not present in any pinned prior Perception media manifest; "
            "SHA256(17:perception-av-event-v1:{official split}:acquisition:{video_id}) order"
        ),
        "official_split_policy": (
            "official TRAIN videos partitioned into disjoint train/calibration groups; "
            "official VALID videos are development validation only; no test archive"
        ),
        "candidate_rule": (
            "one resolved non-Muted source parent, is_visible=1, before split-specific cut; "
            "choose label-blind annotated frame nearest sound interval midpoint among frames "
            "with at least two visible nonempty boxes; hold if source is absent at selected frame"
        ),
        "model_inputs_exclude": [
            "sound label",
            "source parent ID",
            "ground-truth candidate track IDs",
            "video ID",
            "filename",
            "split ID",
        ],
        "participant_disjointness": "UNKNOWN; participant IDs are absent from public annotations",
        "split_limits_bytes": {"per_clip": MAX_CLIP_BYTES, "per_split": MAX_SPLIT_BYTES},
        "splits": {
            "train": {
                "pin": TRAIN_ARCHIVE_PIN,
                "rows": train_rows,
                "candidate_events": sum(row["candidate_events"] for row in train_rows),
                "candidate_videos": len(train_rows),
                "calibration_videos": len(calibration_ids),
                "total_uncompressed_bytes": sum(row["bytes"] for row in train_rows),
            },
            "valid": {
                "pin": VALID_ARCHIVE_PIN,
                "rows": valid_rows,
                "candidate_events": sum(row["candidate_events"] for row in valid_rows),
                "candidate_videos": len(valid_rows),
                "total_uncompressed_bytes": sum(row["bytes"] for row in valid_rows),
            },
        },
        "training": False,
        "model_loaded": False,
        "gpu_used": False,
        "test_opened": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-count", type=int, default=MEDIA_COUNT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required")
    plan = create_plan(media_count=args.media_count)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "plan.json"
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                split: {
                    "videos": entry["candidate_videos"],
                    "events": entry["candidate_events"],
                    "bytes": entry["total_uncompressed_bytes"],
                }
                for split, entry in plan["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
