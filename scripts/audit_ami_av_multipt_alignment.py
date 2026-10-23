"""Audit AMI audio/video timestamp proximity at several points per clip.

This is a metadata/timestamp audit only. It does not inspect speech content,
identify visible speakers, or run a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

EXPECTED_AUDIO_PAIR_RECEIPT_SHA256 = (
    "649e07132e8f7137638a014eb85f7983dc007ddf099824a9fda2453fb5fccdb4"
)
EXPECTED_VIDEO_DECODE_RECEIPT_SHA256 = (
    "757d9514ae9d2957cd975c2edc3715f24cb1383e04a93f777f26ad5ec41b41b2"
)
CAMERAS = ("Closeup1", "Closeup2", "Closeup3", "Closeup4")
SAMPLE_FRACTIONS = (0.1, 0.5, 0.9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_times(begin: float, end: float) -> list[dict[str, float]]:
    if end <= begin:
        raise ValueError("audio interval must have positive duration")
    return [
        {"fraction": fraction, "seconds": begin + (end - begin) * fraction}
        for fraction in SAMPLE_FRACTIONS
    ]


def nearest_video_pts(path: Path, target_seconds: float) -> tuple[float, float, int, int]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to decode RealMedia timestamps") from exc

    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream in {path.name}")
        container.seek(
            int(target_seconds / float(stream.time_base)),
            backward=True,
            any_frame=False,
            stream=stream,
        )
        before: tuple[float, int, int] | None = None
        after: tuple[float, int, int] | None = None
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.time is None:
                    continue
                candidate = (float(frame.time), int(frame.width), int(frame.height))
                if candidate[0] < target_seconds:
                    before = candidate
                else:
                    after = candidate
                    break
            if after is not None:
                break
        candidates = [item for item in (before, after) if item is not None]
        if not candidates:
            raise ValueError(f"no frame near {target_seconds:.6f}s in {path.name}")
        pts, width, height = min(candidates, key=lambda item: abs(item[0] - target_seconds))
        return pts, pts - target_seconds, width, height


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit(
    audio_dir: Path,
    audio_receipt_path: Path,
    video_dir: Path,
    video_manifest_path: Path,
    video_decode_receipt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(audio_receipt_path) != EXPECTED_AUDIO_PAIR_RECEIPT_SHA256:
        raise ValueError("audio-pair receipt hash differs from the pinned preflight")
    if sha256_file(video_decode_receipt_path) != EXPECTED_VIDEO_DECODE_RECEIPT_SHA256:
        raise ValueError("video decode-receipt hash differs from the pinned preflight")

    audio_receipt = _read_json(audio_receipt_path)
    video_manifest = _read_json(video_manifest_path)
    video_decode_receipt = _read_json(video_decode_receipt_path)
    if sha256_file(video_manifest_path) != audio_receipt["video_manifest_sha256"]:
        raise ValueError("video manifest hash does not match audio-pair receipt")
    if sha256_file(video_manifest_path) != video_decode_receipt["manifest_sha256"]:
        raise ValueError("video manifest hash does not match decode receipt")

    paths: dict[tuple[str, str], Path] = {}
    for item in video_manifest.get("files", []):
        path = video_dir / str(item["filename"])
        if path.stat().st_size != int(item["content_length_bytes"]):
            raise ValueError(f"video size mismatch: {path.name}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"video SHA-256 mismatch: {path.name}")
        paths[(str(item["meeting_id"]), f"Closeup{int(item['camera'])}")] = path
    if len(paths) != 24:
        raise ValueError(f"expected 24 verified videos, found {len(paths)}")

    rows = audio_receipt.get("audio_rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("expected the frozen 24-row audio sample")

    started = time.perf_counter()
    audited_rows = []
    for row in rows:
        audio_path = audio_dir / str(row["audio_path"])
        if audio_path.stat().st_size != int(row["audio_bytes"]):
            raise ValueError(f"audio size mismatch: {audio_path.name}")
        if sha256_file(audio_path) != row["audio_sha256"]:
            raise ValueError(f"audio SHA-256 mismatch: {audio_path.name}")
        meeting_id = str(row["meeting_id"])
        if [path.name for path in (paths[(meeting_id, camera)] for camera in CAMERAS)] != row[
            "video_files"
        ]:
            raise ValueError(f"audio/video meeting mapping mismatch: {row['audio_id']}")

        points = []
        for sample in sample_times(float(row["begin_time"]), float(row["end_time"])):
            cameras = []
            for camera in CAMERAS:
                pts, delta, width, height = nearest_video_pts(
                    paths[(meeting_id, camera)], sample["seconds"]
                )
                cameras.append(
                    {
                        "camera": camera,
                        "nearest_pts_seconds": pts,
                        "delta_seconds": delta,
                        "width": width,
                        "height": height,
                    }
                )
            points.append(
                {
                    "fraction": sample["fraction"],
                    "requested_seconds": sample["seconds"],
                    "cameras": cameras,
                }
            )
        audited_rows.append(
            {
                "audio_id": row["audio_id"],
                "meeting_id": meeting_id,
                "experiment_split": row["experiment_split"],
                "speaker_id": row["speaker_id"],
                "sample_points": points,
            }
        )

    deltas = [
        abs(float(camera["delta_seconds"]))
        for row in audited_rows
        for point in row["sample_points"]
        for camera in point["cameras"]
    ]
    output = {
        "schema_version": 1,
        "status": "complete",
        "audit_scope": "audio-video timestamp proximity only",
        "sample_fractions": list(SAMPLE_FRACTIONS),
        "audio_pair_receipt_sha256": sha256_file(audio_receipt_path),
        "video_manifest_sha256": sha256_file(video_manifest_path),
        "video_decode_receipt_sha256": sha256_file(video_decode_receipt_path),
        "audio_clip_count": len(audited_rows),
        "meeting_count": len({row["meeting_id"] for row in audited_rows}),
        "camera_pts_observations": len(deltas),
        "max_abs_pts_delta_seconds": max(deltas),
        "median_abs_pts_delta_seconds": statistics.median(deltas),
        "elapsed_seconds": time.perf_counter() - started,
        "contains_transcripts": False,
        "uses_labels_or_model_predictions": False,
        "audited_rows": audited_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "multipt-alignment-receipt.json"
    fd, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".receipt-", suffix=".partial"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(output, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    output["receipt_sha256"] = sha256_file(destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-receipt", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--video-decode-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.audio_dir,
        args.audio_receipt,
        args.video_dir,
        args.video_manifest,
        args.video_decode_receipt,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "audio_clip_count": result["audio_clip_count"],
                "meeting_count": result["meeting_count"],
                "camera_pts_observations": result["camera_pts_observations"],
                "max_abs_pts_delta_seconds": result["max_abs_pts_delta_seconds"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
