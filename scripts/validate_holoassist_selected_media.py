"""Verify selected HoloAssist train/validation AV files and timing before labeling."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_holoassist_metadata import duration_seconds, sha256_file


def inspect_av_file(path: Path) -> dict:
    import av

    with av.open(str(path)) as container:
        video = [stream for stream in container.streams if stream.type == "video"]
        audio = [stream for stream in container.streams if stream.type == "audio"]
        if len(video) != 1 or len(audio) != 1:
            raise ValueError(f"expected one video and one audio stream: {path}")
        video_stream, audio_stream = video[0], audio[0]
        if video_stream.duration is None or audio_stream.duration is None:
            raise ValueError(f"AV stream duration metadata is missing: {path}")
        video_start = float(video_stream.start_time * video_stream.time_base)
        audio_start = float(audio_stream.start_time * audio_stream.time_base)
        video_duration = float(video_stream.duration * video_stream.time_base)
        audio_duration = float(audio_stream.duration * audio_stream.time_base)
        video_rate = float(
            video_stream.average_rate or video_stream.guessed_rate or video_stream.base_rate or 0
        )
        if video_rate <= 0:
            raise ValueError(f"video frame rate metadata is missing: {path}")
        video_frame_interval = 1.0 / video_rate
        av_duration_tolerance = 2.0 * video_frame_interval
        channels = int(audio_stream.codec_context.channels or 0)
        sample_rate = int(audio_stream.rate or 0)
        if abs(video_start - audio_start) > 0.05:
            raise ValueError(f"audio/video start-time mismatch: {path}")
        av_duration_difference = abs(video_duration - audio_duration)
        if av_duration_difference > av_duration_tolerance:
            raise ValueError(f"audio/video duration mismatch: {path}")
        if channels != 1 or sample_rate != 48_000:
            raise ValueError(f"expected mono 48 kHz audio: {path}")
        return {
            "video_codec": video_stream.codec_context.name,
            "video_start_seconds": video_start,
            "video_duration_seconds": video_duration,
            "video_frame_interval_seconds": video_frame_interval,
            "video_frame_count": int(video_stream.frames or 0),
            "video_width": int(video_stream.codec_context.width),
            "video_height": int(video_stream.codec_context.height),
            "audio_codec": audio_stream.codec_context.name,
            "audio_start_seconds": audio_start,
            "audio_duration_seconds": audio_duration,
            "audio_video_duration_difference_seconds": av_duration_difference,
            "audio_video_duration_tolerance_seconds": av_duration_tolerance,
            "audio_channels": channels,
            "audio_sample_rate_hz": sample_rate,
        }


def validate_selected_media(
    *,
    selection: dict,
    media_manifest: dict,
    annotations: list[dict],
    media_root: Path,
    selection_sha256: str,
    media_manifest_sha256: str,
) -> dict:
    if media_manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("download receipt is for a different immutable selection")
    selected = selection["selection"]
    expected: dict[tuple[str, str], dict] = {}
    for split in ("train", "validation"):
        for row in selected[split]["videos"]:
            key = (split, str(row["session_id"]))
            if key in expected:
                raise ValueError(f"duplicate selected session: {key}")
            expected[key] = row
    train_ids = {session for split, session in expected if split == "train"}
    validation_ids = {session for split, session in expected if split == "validation"}
    if train_ids & validation_ids:
        raise ValueError("selected train/validation sessions overlap")

    media_rows = media_manifest.get("files", [])
    actual = {(str(row["split"]), str(row["session_id"])): row for row in media_rows}
    if len(actual) != len(media_rows) or set(actual) != set(expected):
        raise ValueError("download manifest must exactly cover selected train/validation IDs")
    by_session = {str(row["video_name"]): row for row in annotations}
    if len(by_session) != len(annotations):
        raise ValueError("duplicate annotation session ID")

    reports = []
    for key, selected_row in expected.items():
        split, session_id = key
        item = actual[key]
        if item.get("archive_etag") != selection.get("archive_etag"):
            raise ValueError(f"archive ETag mismatch for {session_id}")
        expected_size = int(selected_row["size_bytes"])
        path = (media_root / split / f"{session_id}.mp4").resolve(strict=True)
        if path.stat().st_size != expected_size or item.get("size_bytes") != expected_size:
            raise ValueError(f"media size mismatch for {session_id}")
        digest = sha256_file(path)
        if digest != item.get("sha256"):
            raise ValueError(f"media SHA-256 mismatch for {session_id}")
        annotation = by_session.get(session_id)
        if annotation is None:
            raise ValueError(f"missing train/validation annotation for {session_id}")
        annotation_duration = duration_seconds(
            annotation.get("videoMetadata", {}).get("duration", {}).get("raw")
        )
        if annotation_duration is None:
            raise ValueError(f"invalid annotation duration for {session_id}")
        facts = inspect_av_file(path)
        if abs(facts["audio_duration_seconds"] - annotation_duration) > 0.15:
            raise ValueError(f"audio/annotation duration mismatch for {session_id}")
        reports.append(
            {
                "split": split,
                "session_id": session_id,
                "media_path": f"{split}/{session_id}.mp4",
                "media_size_bytes": expected_size,
                "media_sha256": digest,
                "annotation_duration_seconds": annotation_duration,
                **facts,
            }
        )

    return {
        "status": "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA",
        "selection_sha256": selection_sha256,
        "media_manifest_sha256": media_manifest_sha256,
        "train_session_count": len(train_ids),
        "validation_session_count": len(validation_ids),
        "audio_annotation_duration_difference_seconds_median": statistics.median(
            abs(row["audio_duration_seconds"] - row["annotation_duration_seconds"])
            for row in reports
        ),
        "audio_video_duration_difference_seconds_max": max(
            abs(row["audio_duration_seconds"] - row["video_duration_seconds"])
            for row in reports
        ),
        "audio_video_duration_difference_max_video_frames": max(
            row["audio_video_duration_difference_seconds"]
            / row["video_frame_interval_seconds"]
            for row in reports
        ),
        "media": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite AV validation report: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    media_manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotation.read_text(encoding="utf-8"))
    selection_hash = sha256_file(args.selection)
    media_hash = sha256_file(args.media_manifest)
    report = validate_selected_media(
        selection=selection,
        media_manifest=media_manifest,
        annotations=annotations,
        media_root=args.media_root,
        selection_sha256=selection_hash,
        media_manifest_sha256=media_hash,
    )
    report.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "annotation_sha256": sha256_file(args.annotation),
            "media_root": str(args.media_root.resolve(strict=True)),
            "test_labels_opened": False,
            "test_media_opened": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"media", "created_at_utc"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
