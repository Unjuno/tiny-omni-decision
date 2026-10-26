"""Freeze a fixed, official-validation source-candidate transfer cohort."""

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

from tiny_omni_decision.perception_source_candidate import (
    consolidate_source_candidate_records,
    source_candidate_records,
)

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
EXPECTED_VALIDATION_SHA256 = "4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141"
EXPECTED_CUT_MAP_SHA256 = "3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7"
VALIDATION_MEMBER = "challenge_sound_localisation_valid.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_selection(selection):
    events = selection.get("events")
    if (
        selection.get("source_split") != "official validation only"
        or selection.get("official_test_split_opened") is not False
        or not isinstance(events, list)
        or not events
    ):
        raise ValueError("selection must be a nonempty official-validation cohort")
    video_ids = [str(event.get("video_id", "")) for event in events]
    event_keys = [
        (str(event.get("video_id", "")), str(event.get("event_id", "")))
        for event in events
    ]
    if (
        any(not value for value in video_ids)
        or any(not event_id.isdecimal() for _, event_id in event_keys)
        or len(video_ids) != len(set(video_ids))
        or len(event_keys) != len(set(event_keys))
        or selection.get("selected_video_count") != len(events)
    ):
        raise ValueError("selection must contain exactly one event per validation video")
    if any(event.get("split") != "valid" for event in events):
        raise ValueError("selection contains a non-validation event")
    return events, set(video_ids)


def build(annotation_zip, cut_map_path, selection_path, output_dir):
    annotation_zip = Path(annotation_zip)
    cut_map_path = Path(cut_map_path)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior validation corpus: {output_dir}")

    annotation_sha = sha256(annotation_zip)
    if annotation_sha != EXPECTED_VALIDATION_SHA256:
        raise ValueError("validation annotation archive hash differs from pinned source")
    cut_map_sha = sha256(cut_map_path)
    if cut_map_sha != EXPECTED_CUT_MAP_SHA256:
        raise ValueError("validation cut-map hash differs from pinned source")
    selection_sha = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    events, selected_videos = _validate_selection(selection)

    cut_map = json.loads(cut_map_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(annotation_zip) as archive:
        if VALIDATION_MEMBER not in archive.namelist():
            raise ValueError("pinned source archive lacks the official validation member")
        source = json.loads(archive.read(VALIDATION_MEMBER))
    source_video_ids = {str(row["metadata"]["video_id"]) for row in source.values()}
    cut_video_ids = set(cut_map) - {"nid"}
    if not source_video_ids <= cut_video_ids:
        raise ValueError("validation cut-map omits annotation video IDs")
    if not selected_videos <= source_video_ids:
        raise ValueError("selection includes videos absent from pinned validation annotations")

    selected_source_rows = [
        row
        for row in source.values()
        if str(row["metadata"]["video_id"]) in selected_videos
    ]
    videos_without_tracks = sorted(
        str(row["metadata"]["video_id"])
        for row in selected_source_rows
        if not isinstance(row.get("object_tracking"), list)
        or not row["object_tracking"]
    )
    if videos_without_tracks:
        raise ValueError(
            "official validation source-candidate transfer requires usable "
            "object_tracking annotations; missing or empty for "
            f"{len(videos_without_tracks)} selected video(s), including "
            f"{', '.join(videos_without_tracks[:5])}"
        )

    selected_events = {
        f"{event['video_id']}:sound:{int(event['event_id'])}": str(event["video_id"])
        for event in events
    }
    rows_by_selected_video = collections.defaultdict(list)
    for row in sorted(source.values(), key=lambda item: str(item["metadata"]["video_id"])):
        video_id = str(row["metadata"]["video_id"])
        if video_id not in selected_videos:
            continue
        raw_records = source_candidate_records(
            video_id,
            row,
            cut_frame=cut_map[video_id],
            expected_split="valid",
        )
        rows_by_selected_video[video_id].extend(
            consolidate_source_candidate_records(raw_records)
        )

    by_id = {
        record["record_id"]: record
        for video_records in rows_by_selected_video.values()
        for record in video_records
    }
    if len(by_id) != sum(map(len, rows_by_selected_video.values())):
        raise ValueError("selected validation cohort contains duplicate source event IDs")
    missing_events = sorted(set(selected_events) - set(by_id))
    if missing_events:
        raise ValueError("selected events are absent from source-candidate annotations")
    if any(
        by_id[event_id]["group_id"] != video_id
        for event_id, video_id in selected_events.items()
    ):
        raise ValueError("selected event/video identities disagree with annotations")

    selected_rows = [
        by_id[f"{event['video_id']}:sound:{int(event['event_id'])}"]
        for event in events
    ]
    if {row["group_id"] for row in selected_rows} != selected_videos:
        raise ValueError("selected source-candidate rows do not preserve video grouping")
    status_counts = collections.Counter(row["status"] for row in selected_rows)
    output_dir.mkdir(parents=True)
    record_path = output_dir / "validation_transfer_records.jsonl"
    with record_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected_rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "format": "perception_test_source_candidate_validation_transfer_v1",
        "source_revision": SOURCE_REVISION,
        "source_split": "official_validation",
        "annotation_member": VALIDATION_MEMBER,
        "annotation_archive_sha256": annotation_sha,
        "cut_map_sha256": cut_map_sha,
        "selection_sha256": selection_sha,
        "selection_policy": (
            "fixed one-event-per-video cohort selected before this task from official validation; "
            "visibility-balanced in a separate task; no source-candidate model outcomes used"
        ),
        "video_group_count": len(selected_videos),
        "selected_event_count": len(selected_rows),
        "selected_status_counts": dict(sorted(status_counts.items())),
        "eligible_candidate_count": status_counts["candidate"],
        "official_test_split_opened": False,
        "sealed_audit_opened": False,
        "model_loaded": False,
        "metrics_computed": False,
        "record_file": record_path.name,
        "record_file_sha256": sha256(record_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--validation-cut-map", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.validation_annotations,
                args.validation_cut_map,
                args.selection,
                args.output_dir,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
