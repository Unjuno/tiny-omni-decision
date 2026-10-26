"""Freeze label-free media, audio-window, and ROI mappings before encoder runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tiny_omni_decision.perception_audio_windows import split_audio_interval
from tiny_omni_decision.perception_video_roi import map_bbox_to_token_grid
from tiny_omni_decision.perception_video_windows import plan_frame_windows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rows(records, split_name):
    ids = [row["record_id"] for row in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError(f"{split_name} rows must be nonempty with unique record IDs")
    for row in records:
        if row.get("status") != "candidate" or row.get("split") != "train":
            raise ValueError("feature plans accept only candidate rows from official train")
        if not row["request"].get("candidates"):
            raise ValueError(f"candidate list is empty for {row['record_id']}")


def _validate_control(records, control, expected_sha, split_name):
    if control.get("source_jsonl_sha256") != expected_sha:
        raise ValueError(f"{split_name} control source hash mismatch")
    by_id = {row["record_id"]: row for row in control["records"]}
    record_ids = {row["record_id"] for row in records}
    if len(by_id) != len(control["records"]) or set(by_id) != record_ids:
        raise ValueError(f"{split_name} control record IDs differ from fixed source split")
    return by_id


def _validate_media_report(records, report, clips, expected_sha, acquisition_sha, split_name):
    if report.get("status") not in {"PASS", "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR"}:
        raise ValueError(f"{split_name} media report is not a passing preflight")
    if report.get("selection") != "explicit_ordered_record_ids":
        raise ValueError(f"{split_name} media report is not for the explicit fixed split")
    if report.get("selection_source_jsonl_sha256") != expected_sha:
        raise ValueError(f"{split_name} media report source hash mismatch")
    if report.get("acquisition_manifest_sha256") != acquisition_sha:
        raise ValueError(f"{split_name} media report acquisition hash mismatch")
    report_rows = {row["record_id"]: row for row in report.get("records", [])}
    expected_rows = {row["record_id"]: row for row in records}
    if len(report_rows) != len(report.get("records", [])) or set(report_rows) != set(expected_rows):
        raise ValueError(f"{split_name} media report IDs differ from fixed source split")
    for record_id, source_row in expected_rows.items():
        report_row = report_rows[record_id]
        if report_row.get("video_id") != source_row["group_id"]:
            raise ValueError(f"{split_name} media report video mismatch for {record_id}")
        clip = clips.get(source_row["group_id"])
        if clip is None or report_row.get("media_sha256") != clip.get("sha256"):
            raise ValueError(f"{split_name} media report hash mismatch for {record_id}")
        repair = report_row.get("terminal_frame_repair")
        if repair is not None:
            candidates = {
                candidate["candidate_id"]: candidate
                for candidate in source_row["request"]["candidates"]
            }
            if repair.get("record_id") != record_id:
                raise ValueError(f"{split_name} frame repair record mismatch")
            repaired_candidates = set(repair.get("candidate_ids", []))
            if not repaired_candidates or not repaired_candidates <= set(candidates):
                raise ValueError(f"{split_name} frame repair has invalid candidate IDs")
            if repair.get("declared_frame_count") != repair.get(
                "decoded_frame_count", -1
            ) + 1 or repair.get("frame_id") != repair.get("decoded_frame_count"):
                raise ValueError(f"{split_name} frame repair is not a terminal-frame correction")
            for candidate_id, candidate in candidates.items():
                frame_ids = [box["frame_id"] for box in candidate["frame_boxes"]]
                retained = sum(frame_id != repair["frame_id"] for frame_id in frame_ids)
                if retained < 1:
                    raise ValueError(f"frame repair leaves candidate {candidate_id} with no frames")
                if repair["retained_frame_count_by_candidate"].get(candidate_id) != retained:
                    raise ValueError(f"frame repair retained-count mismatch for {candidate_id}")
                affected = repair["frame_id"] in frame_ids
                if affected != (candidate_id in repair["candidate_ids"]):
                    raise ValueError(f"frame repair candidate mapping mismatch for {candidate_id}")
    if report.get("selected_record_count") != len(expected_rows):
        raise ValueError(f"{split_name} media report selected count mismatch")
    has_repairs = any(row.get("terminal_frame_repair") for row in report_rows.values())
    if (report["status"] == "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR") != has_repairs:
        raise ValueError(f"{split_name} media report repair status is inconsistent")
    return report_rows


def _build_split_plan(records, clips, control, expected_sha, split_name, media_rows):
    _validate_rows(records, split_name)
    control_rows = _validate_control(records, control, expected_sha, split_name)
    rows_by_video = defaultdict(list)
    frames_by_video = defaultdict(set)
    for row in records:
        video_id = row["group_id"]
        if video_id not in clips:
            raise ValueError(f"missing local acquisition receipt for video {video_id}")
        clip = clips[video_id]
        metadata = clip["metadata"]
        if metadata.get("split") != "train" or metadata.get("video_id") != video_id:
            raise ValueError(f"non-train or mismatched acquisition receipt for {video_id}")
        resolution = metadata.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
        ):
            raise ValueError(f"invalid source resolution for {video_id}")
        rows_by_video[video_id].append(row)
        for candidate in row["request"]["candidates"]:
            for box in candidate["frame_boxes"]:
                repair = media_rows[row["record_id"]].get("terminal_frame_repair")
                if repair is None or box["frame_id"] != repair["frame_id"]:
                    frames_by_video[video_id].add(box["frame_id"])

    video_plans = {}
    for video_id in sorted(rows_by_video):
        clip = clips[video_id]
        height, width = clip["metadata"]["resolution"]
        plan = plan_frame_windows(sorted(frames_by_video[video_id]))
        video_plans[video_id] = {
            "video_id": video_id,
            "media_sha256": clip["sha256"],
            "media_bytes": clip["bytes"],
            "source_resolution_hw": [height, width],
            "frame_ids": sorted(frames_by_video[video_id]),
            **plan,
        }

    serialized_records = []
    for row in sorted(records, key=lambda item: item["record_id"]):
        video_id = row["group_id"]
        height, width = clips[video_id]["metadata"]["resolution"]
        frame_to_slot = video_plans[video_id]["frame_to_window_slot"]
        control_row = control_rows[row["record_id"]]
        repair = media_rows[row["record_id"]].get("terminal_frame_repair")
        source_interval = row["request"]["audio_interval_us"]
        shifted_interval = control_row["offset_audio_interval_us"]
        candidates = []
        for candidate in row["request"]["candidates"]:
            frames = []
            for box in candidate["frame_boxes"]:
                frame_id = box["frame_id"]
                if repair is not None and frame_id == repair["frame_id"]:
                    continue
                frames.append(
                    {
                        "frame_id": frame_id,
                        "window_slot": list(frame_to_slot[frame_id]),
                        "roi_xyxy_tokens": list(
                            map_bbox_to_token_grid(box["bbox_xyxy_normalized"], width, height)
                        ),
                    }
                )
            if not frames:
                raise ValueError(f"frame repair leaves candidate {candidate['candidate_id']} empty")
            candidates.append({"candidate_id": candidate["candidate_id"], "frames": frames})

        audio_windows = split_audio_interval(*source_interval)
        shifted_audio_windows = (
            split_audio_interval(*shifted_interval) if shifted_interval is not None else None
        )
        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        if set(control_row["candidate_order"]) != candidate_ids:
            raise ValueError(f"candidate order mismatch for {row['record_id']}")
        serialized_records.append(
            {
                "record_id": row["record_id"],
                "video_id": video_id,
                "terminal_frame_repair": repair,
                "audio_windows": audio_windows,
                "shifted_audio_interval_us": shifted_interval,
                "shifted_audio_windows": shifted_audio_windows,
                "audio_shuffle_donor_record_id": control_row["audio_shuffle_donor_record_id"],
                "audio_shuffle_donor_video_id": control_row["audio_shuffle_donor_video_id"],
                "candidate_order": control_row["candidate_order"],
                "candidates": candidates,
            }
        )

    clip_hashes = {clips[video_id]["sha256"] for video_id in rows_by_video}
    return {
        "record_count": len(serialized_records),
        "video_group_count": len(video_plans),
        "unique_annotated_frame_count": sum(
            len(plan["frame_ids"]) for plan in video_plans.values()
        ),
        "videoprism_clip_count": sum(len(plan["windows"]) for plan in video_plans.values()),
        "source_jsonl_sha256": expected_sha,
        "terminal_frame_repairs": [
            media_rows[row["record_id"]]["terminal_frame_repair"]
            for row in records
            if media_rows[row["record_id"]].get("terminal_frame_repair") is not None
        ],
        "media_sha256": sorted(clip_hashes),
        "videos": list(video_plans.values()),
        "records": serialized_records,
    }


def build_validation_feature_plan(
    records,
    clips,
    media_report,
    *,
    record_jsonl_sha256,
    acquisition_manifest_sha256,
    media_report_sha256,
    selection_sha256,
):
    """Build a label-free plan for the fixed, non-blind official-validation cohort."""
    ids = [row.get("record_id") for row in records]
    videos = [row.get("group_id") for row in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("official validation rows must be nonempty with unique record IDs")
    if len(videos) != len(set(videos)):
        raise ValueError("official validation cohort must contain one event per video")
    for row in records:
        if (
            row.get("status") != "candidate"
            or row.get("split") != "validation"
            or row.get("provenance", {}).get("source_split") != "official_validation"
        ):
            raise ValueError("rows must come from the fixed official validation split")
        if not row.get("request", {}).get("candidates"):
            raise ValueError(f"candidate list is empty for {row['record_id']}")

    if media_report.get("status") not in {
        "PASS",
        "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR",
    }:
        raise ValueError("official validation media report did not pass")
    if media_report.get("selection") != "explicit_ordered_record_ids":
        raise ValueError("official validation media report is not bound to fixed record IDs")
    if media_report.get("selection_source_jsonl_sha256") != record_jsonl_sha256:
        raise ValueError("official validation media report source hash mismatch")
    if media_report.get("acquisition_manifest_sha256") != acquisition_manifest_sha256:
        raise ValueError("official validation acquisition hash mismatch")
    report_rows = {row.get("record_id"): row for row in media_report.get("records", [])}
    if len(report_rows) != len(media_report.get("records", [])) or set(report_rows) != set(ids):
        raise ValueError("official validation media report IDs differ from frozen corpus")

    rows_by_video = {}
    frames_by_video = defaultdict(set)
    for row in records:
        video_id = row["group_id"]
        clip = clips.get(video_id)
        if clip is None:
            raise ValueError(f"missing official validation media receipt for {video_id}")
        metadata = clip.get("metadata", {})
        if metadata.get("split") != "validation" or metadata.get("video_id") != video_id:
            raise ValueError(f"non-validation or mismatched acquisition receipt for {video_id}")
        resolution = metadata.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
        ):
            raise ValueError(f"invalid source resolution for {video_id}")
        checked = report_rows[row["record_id"]]
        if checked.get("video_id") != video_id or checked.get("media_sha256") != clip.get("sha256"):
            raise ValueError(f"official validation media receipt mismatch for {video_id}")
        decoded = set(checked.get("decoded_frame_ids", []))
        repair = checked.get("terminal_frame_repair")
        if repair is not None and repair.get("record_id") != row["record_id"]:
            raise ValueError("official validation terminal-frame repair record mismatch")
        rows_by_video[video_id] = row
        for candidate in row["request"]["candidates"]:
            candidate_frames = []
            for box in candidate["frame_boxes"]:
                frame_id = box["frame_id"]
                if repair is not None and frame_id == repair.get("frame_id"):
                    continue
                if frame_id not in decoded:
                    raise ValueError(
                        f"requested frame absent from validation preflight: {row['record_id']}"
                    )
                frames_by_video[video_id].add(frame_id)
                candidate_frames.append(frame_id)
            if not candidate_frames:
                raise ValueError(f"no preflighted candidate frames remain for {row['record_id']}")

    video_plans = []
    frame_slots = {}
    for video_id in sorted(rows_by_video):
        clip = clips[video_id]
        height, width = clip["metadata"]["resolution"]
        frame_ids = sorted(frames_by_video[video_id])
        window_plan = plan_frame_windows(frame_ids)
        frame_slots[video_id] = window_plan["frame_to_window_slot"]
        video_plans.append(
            {
                "video_id": video_id,
                "media_sha256": clip["sha256"],
                "media_bytes": clip["bytes"],
                "source_resolution_hw": [height, width],
                "frame_ids": frame_ids,
                **window_plan,
            }
        )

    serialized_records = []
    for row in sorted(records, key=lambda item: item["record_id"]):
        video_id = row["group_id"]
        height, width = clips[video_id]["metadata"]["resolution"]
        repair = report_rows[row["record_id"]].get("terminal_frame_repair")
        candidates = []
        for candidate in row["request"]["candidates"]:
            frames = []
            for box in candidate["frame_boxes"]:
                frame_id = box["frame_id"]
                if repair is not None and frame_id == repair.get("frame_id"):
                    continue
                frames.append(
                    {
                        "frame_id": frame_id,
                        "window_slot": list(frame_slots[video_id][frame_id]),
                        "roi_xyxy_tokens": list(
                            map_bbox_to_token_grid(box["bbox_xyxy_normalized"], width, height)
                        ),
                    }
                )
            candidates.append({"candidate_id": candidate["candidate_id"], "frames": frames})
        serialized_records.append(
            {
                "record_id": row["record_id"],
                "video_id": video_id,
                "upstream_cut_frame": row["provenance"].get("upstream_cut_frame"),
                "terminal_frame_repair": repair,
                "audio_windows": split_audio_interval(*row["request"]["audio_interval_us"]),
                "shifted_audio_windows": None,
                "candidates": candidates,
            }
        )

    return {
        "format": "perception_test_source_candidate_validation_feature_plan_v1",
        "seed": 17,
        "purpose": "fixed_checkpoint_nonblind_validation_transfer_only",
        "label_free": True,
        "model_loaded": False,
        "metrics_computed": False,
        "selection_sha256": selection_sha256,
        "source_jsonl_sha256": record_jsonl_sha256,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "validation_media_report_sha256": media_report_sha256,
        "video_encoder": {
            "name": "VideoPrism-B",
            "clip_frames": 8,
            "clip_stride": 4,
            "letterbox_frame_size": 288,
            "token_grid_size": 16,
            "token_dimension": 768,
        },
        "audio_encoder": {
            "name": "EfficientAT",
            "sample_rate_hz": 32_000,
            "window_seconds": 2,
            "window_policy": "nonoverlap; centered zero pad; event interval exactly once",
            "feature_dimension": 384,
        },
        "validation": {
            "record_count": len(serialized_records),
            "video_group_count": len(video_plans),
            "upstream_cut_frame_records": sum(
                row["provenance"].get("upstream_cut_frame", -1) >= 0 for row in records
            ),
            "unique_annotated_frame_count": sum(len(row["frame_ids"]) for row in video_plans),
            "videoprism_clip_count": sum(len(row["windows"]) for row in video_plans),
            "media_sha256": sorted({clips[video_id]["sha256"] for video_id in rows_by_video}),
            "videos": video_plans,
            "records": serialized_records,
        },
    }


def prepare_feature_plan(
    train_records,
    development_records,
    clips,
    train_control,
    development_control,
    *,
    train_jsonl_sha256,
    development_jsonl_sha256,
    acquisition_manifest_sha256,
    train_media_report,
    development_media_report,
    train_media_report_sha256,
    development_media_report_sha256,
):
    """Build the fixed label-free feature plan from immutable split/control inputs."""
    train_groups = {row["group_id"] for row in train_records}
    development_groups = {row["group_id"] for row in development_records}
    if train_groups & development_groups:
        raise ValueError("split video groups overlap")

    train_media_rows = _validate_media_report(
        train_records,
        train_media_report,
        clips,
        train_jsonl_sha256,
        acquisition_manifest_sha256,
        "train",
    )
    development_media_rows = _validate_media_report(
        development_records,
        development_media_report,
        clips,
        development_jsonl_sha256,
        acquisition_manifest_sha256,
        "development",
    )
    train = _build_split_plan(
        train_records, clips, train_control, train_jsonl_sha256, "train", train_media_rows
    )
    development = _build_split_plan(
        development_records,
        clips,
        development_control,
        development_jsonl_sha256,
        "development",
        development_media_rows,
    )
    if set(train["media_sha256"]) & set(development["media_sha256"]):
        raise ValueError("split media hashes overlap")
    return {
        "format": "perception_test_source_candidate_feature_plan_v1",
        "seed": 17,
        "label_free": True,
        "model_loaded": False,
        "metrics_computed": False,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "train_media_report_sha256": train_media_report_sha256,
        "development_media_report_sha256": development_media_report_sha256,
        "video_encoder": {
            "name": "VideoPrism-B",
            "clip_frames": 8,
            "clip_stride": 4,
            "letterbox_frame_size": 288,
            "token_grid_size": 16,
            "token_dimension": 768,
        },
        "audio_encoder": {
            "name": "EfficientAT",
            "sample_rate_hz": 32_000,
            "window_seconds": 2,
            "window_policy": "nonoverlap; centered zero pad; entire event interval exactly once",
            "feature_dimension": 384,
        },
        "train": train,
        "development": development,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--development-jsonl", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--train-media-report", type=Path, required=True)
    parser.add_argument("--development-media-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen feature plan: {args.output}")

    def read_rows(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    acquisition = json.loads(args.acquisition_manifest.read_text(encoding="utf-8"))
    control = json.loads(args.control_manifest.read_text(encoding="utf-8"))
    train_media_report = json.loads(args.train_media_report.read_text(encoding="utf-8"))
    development_media_report = json.loads(args.development_media_report.read_text(encoding="utf-8"))
    if control.get("seed") != 17:
        raise ValueError("fixed feature plan requires seed-17 controls")
    clips = {}
    for clip in acquisition["clips"]:
        video_id = clip["metadata"]["video_id"]
        if video_id in clips:
            raise ValueError(f"duplicate acquisition receipt for {video_id}")
        clips[video_id] = clip

    plan = prepare_feature_plan(
        read_rows(args.train_jsonl),
        read_rows(args.development_jsonl),
        clips,
        control["train"],
        control["development"],
        train_jsonl_sha256=sha256_file(args.train_jsonl),
        development_jsonl_sha256=sha256_file(args.development_jsonl),
        acquisition_manifest_sha256=sha256_file(args.acquisition_manifest),
        train_media_report=train_media_report,
        development_media_report=development_media_report,
        train_media_report_sha256=sha256_file(args.train_media_report),
        development_media_report_sha256=sha256_file(args.development_media_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    summary = {
        "format": plan["format"],
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "model_loaded": plan["model_loaded"],
        "metrics_computed": plan["metrics_computed"],
        "train": {
            key: plan["train"][key]
            for key in (
                "record_count",
                "video_group_count",
                "unique_annotated_frame_count",
                "videoprism_clip_count",
            )
        },
        "development": {
            key: plan["development"][key]
            for key in (
                "record_count",
                "video_group_count",
                "unique_annotated_frame_count",
                "videoprism_clip_count",
            )
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
