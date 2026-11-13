"""Verify CATER media hashes and count source video frames for temporal gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_media(media_root: Path, media_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import av

    records = []
    seen = set()
    for source in media_records:
        split = str(source["split"])
        video_id = str(source["video_id"])
        key = (split, video_id)
        if key in seen:
            raise ValueError(f"duplicate media record: {key}")
        seen.add(key)
        relative = Path(str(source["relative_path"]))
        path = (media_root / relative).resolve(strict=True)
        if media_root.resolve(strict=True) not in path.parents:
            raise ValueError(f"media path escapes the acquisition root: {relative}")
        if path.stat().st_size != int(source["media_bytes"]):
            raise ValueError(f"media size mismatch: {path}")
        media_hash = sha256_file(path)
        if media_hash != source["media_sha256"]:
            raise ValueError(f"media SHA-256 mismatch: {path}")
        with av.open(str(path)) as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if len(streams) != 1:
                raise ValueError(f"expected exactly one video stream: {path}")
            stream = streams[0]
            declared_frames = int(stream.frames or 0)
            first_pts = None
            last_pts = None
            decoded_frames = 0
            decoded_packet_pts = set()
            invalid_packet_pts = set()
            for packet in container.demux(stream):
                if packet.pts is None or packet.size <= 0:
                    continue
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    invalid_packet_pts.add(int(packet.pts))
                    continue
                if not frames:
                    invalid_packet_pts.add(int(packet.pts))
                    continue
                decoded_packet_pts.add(int(packet.pts))
                for frame in frames:
                    decoded_frames += 1
                    if first_pts is None:
                        first_pts = frame.pts
                    last_pts = frame.pts
            missing_packet_pts = sorted(
                set(range(declared_frames)) - decoded_packet_pts
            ) if declared_frames else []
            unavailable_packet_pts = sorted(invalid_packet_pts | set(missing_packet_pts))
            decode_status = "PASS" if not unavailable_packet_pts else "INCOMPLETE"
            records.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "media_sha256": media_hash,
                    "media_bytes": path.stat().st_size,
                    "stream_frames": declared_frames,
                    "decoded_frames": decoded_frames,
                    "decoded_packet_pts": sorted(decoded_packet_pts),
                    "unavailable_packet_pts": unavailable_packet_pts,
                    "decode_status": decode_status,
                    "codec": getattr(
                        getattr(stream, "codec_context", None), "name", None
                    ),
                    "fps": str(stream.average_rate),
                    "duration_frames": stream.duration,
                    "first_pts": first_pts,
                    "last_pts": last_pts,
                }
            )
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    media_dir = args.media_dir.resolve(strict=True)
    media_manifest = args.media_manifest.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite media frame audit: {output}")
    rows = _read_jsonl(media_manifest)
    records = audit_media(media_dir, rows)
    output.mkdir(parents=True)
    record_path = output / "records.jsonl"
    record_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    frame_counts = Counter(int(row["decoded_frames"]) for row in records)
    complete = all(row["decode_status"] == "PASS" for row in records)
    summary = {
        "status": (
            "PASS_CATER_MEDIA_FRAME_AUDIT"
            if complete
            else "CATER_MEDIA_DECODE_ISSUES_FOUND"
        ),
        "media_root": str(media_dir),
        "media_manifest_sha256": sha256_file(media_manifest),
        "media_count": len(records),
        "counts_by_split": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "validation")
        },
        "frame_count_histogram": dict(sorted(frame_counts.items())),
        "below_301_frame_count": sum(
            count for frames, count in frame_counts.items() if frames < 301
        ),
        "incomplete_video_count": sum(row["decode_status"] != "PASS" for row in records),
        "records_sha256": sha256_file(record_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
