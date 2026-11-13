"""Verify the fixed non-blind validation cohort and its requested media frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(records, corpus_manifest, acquisition, media_validation, media_root: Path):
    import av

    selection_sha = corpus_manifest["selection_sha256"]
    if (
        corpus_manifest.get("source_split") != "official_validation"
        or corpus_manifest.get("official_test_split_opened")
        or corpus_manifest.get("sealed_audit_opened")
        or media_validation.get("selection_sha256") != selection_sha
        or acquisition.get("selection_sha256") != selection_sha
    ):
        raise ValueError("validation corpus/media provenance does not match the fixed selection")
    if media_validation.get("status") != "PASS_MEDIA_HASH_AND_DECODE_METADATA":
        raise ValueError("the acquired media validation receipt is not a pass")
    files = {}
    for row in acquisition.get("files", []):
        key = (row.get("video_id"), Path(row.get("member", "")).suffix.lower())
        if key in files:
            raise ValueError(f"duplicate acquired media receipt: {key}")
        files[key] = row
    media_rows = {row["video_id"]: row for row in media_validation["examples_manifest"]}
    if len(media_rows) != len(media_validation["examples_manifest"]):
        raise ValueError("media validation receipt contains duplicate video IDs")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("validation corpus contains duplicate record IDs")
    if len({row["group_id"] for row in records}) != len(records):
        raise ValueError("fixed validation cohort must contain one event per video")

    clips, report_rows = [], []
    for record in sorted(records, key=lambda row: row["record_id"]):
        video_id = record["group_id"]
        if (
            record.get("split") != "validation"
            or record.get("status") != "candidate"
            or record.get("provenance", {}).get("source_split") != "official_validation"
        ):
            raise ValueError(f"unexpected non-official validation row: {record['record_id']}")
        expected = media_rows.get(video_id)
        video_receipt = files.get((video_id, ".mp4"))
        audio_receipt = files.get((video_id, ".wav"))
        if expected is None or video_receipt is None or audio_receipt is None:
            raise ValueError(f"missing validation media receipt for {video_id}")
        video_path = media_root / f"{video_id}.mp4"
        audio_path = media_root / f"{video_id}.wav"
        for path, receipt, expected_sha in (
            (video_path, video_receipt, expected["video_sha256"]),
            (audio_path, audio_receipt, expected["audio_sha256"]),
        ):
            if not path.is_file() or path.stat().st_size != receipt["bytes"]:
                raise ValueError(f"validation media file is missing or has wrong size: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != receipt["sha256"] or actual_sha != expected_sha:
                raise ValueError(f"validation media file hash mismatch: {path}")

        wanted = {
            box["frame_id"]
            for candidate in record["request"]["candidates"]
            for box in candidate["frame_boxes"]
        }
        if not wanted or min(wanted) < 0:
            raise ValueError(f"invalid requested frame IDs: {record['record_id']}")
        decoded = set()
        with av.open(str(video_path)) as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if len(video_streams) != 1 or not audio_streams:
                raise ValueError(f"validation MP4 lacks a unique video/audio stream: {video_id}")
            stream = video_streams[0]
            height, width = stream.codec_context.height, stream.codec_context.width
            if min(height, width) <= 0:
                raise ValueError(f"invalid video dimensions: {video_id}")
            for frame_id, _frame in enumerate(container.decode(stream)):
                if frame_id in wanted:
                    decoded.add(frame_id)
                if frame_id >= max(wanted):
                    break
            rate = float(stream.average_rate) if stream.average_rate else None
            declared_frames = int(stream.frames) if stream.frames else None
            audio_stream = audio_streams[0]
            audio_rate = int(audio_stream.codec_context.sample_rate or 0)
            audio_channels = int(audio_stream.codec_context.channels or 0)
            duration_us = int(container.duration or 0) * 1_000_000 // av.time_base
        if decoded != wanted:
            raise ValueError(
                f"requested validation frames unavailable for {video_id}: "
                f"missing={sorted(wanted - decoded)}"
            )
        event_start, event_end = record["request"]["audio_interval_us"]
        if event_start < 0 or event_end <= event_start or event_end > duration_us:
            raise ValueError(
                f"validation event interval outside source duration: {record['record_id']}"
            )
        clip = {
            "metadata": {
                "video_id": video_id,
                "split": "validation",
                "resolution": [height, width],
                "frame_rate": rate,
                "num_frames": declared_frames,
                "audio_sample_rate": audio_rate,
                "audio_channels": audio_channels,
                "duration_us": duration_us,
            },
            "sha256": video_receipt["sha256"],
            "bytes": video_receipt["bytes"],
            "audio_sha256": audio_receipt["sha256"],
            "audio_bytes": audio_receipt["bytes"],
        }
        clips.append(clip)
        report_rows.append(
            {
                "record_id": record["record_id"],
                "video_id": video_id,
                "upstream_cut_frame": record["provenance"].get("upstream_cut_frame"),
                "media_sha256": video_receipt["sha256"],
                "audio_sha256": audio_receipt["sha256"],
                "decoded_frame_ids": sorted(decoded),
                "terminal_frame_repair": None,
            }
        )
    return {
        "acquisition": {
            "format": "perception_test_source_candidate_validation_media_v1",
            "selection_sha256": selection_sha,
            "source_acquisition_manifest_sha256": hashlib.sha256(
                json.dumps(acquisition, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "clips": clips,
        },
        "report": {
            "status": "PASS",
            "selection": "explicit_ordered_record_ids",
            "selection_sha256": selection_sha,
            "selected_record_count": len(records),
            "official_test_split_opened": False,
            "sealed_audit_opened": False,
            "records": report_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--media-validation", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite validation preflight: {args.output_dir}")
    corpus_rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [row for row in corpus_rows if row.get("status") == "candidate"]
    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    acquisition = json.loads(args.acquisition_manifest.read_text(encoding="utf-8"))
    media_validation = json.loads(args.media_validation.read_text(encoding="utf-8"))
    if hashlib.sha256(args.corpus.read_bytes()).hexdigest() != manifest["record_file_sha256"]:
        raise ValueError("fixed official validation corpus hash mismatch")
    if len(records) != manifest.get("eligible_candidate_count"):
        raise ValueError("candidate record count differs from frozen validation manifest")
    result = preflight(records, manifest, acquisition, media_validation, args.media_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    acquisition_path = args.output_dir / "acquisition-manifest.json"
    acquisition_path.write_text(
        json.dumps(result["acquisition"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = result["report"]
    report["selection_source_jsonl_sha256"] = manifest["record_file_sha256"]
    report["acquisition_manifest_sha256"] = sha256_file(acquisition_path)
    report_path = args.output_dir / "media-preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "record_count": len(records),
                "video_count": len(result["acquisition"]["clips"]),
                "acquisition_manifest_sha256": report["acquisition_manifest_sha256"],
                "media_report_sha256": sha256_file(report_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
