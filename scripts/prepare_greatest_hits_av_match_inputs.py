"""Decode fixed audio/video assets, audit full-content overlap, and freeze inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

NUM_VIDEO_FRAMES = 8
FRAME_SIZE = (288, 288)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_frame_indices(frame_count: int, sample_count: int = NUM_VIDEO_FRAMES) -> list[int]:
    if frame_count < sample_count or sample_count < 2:
        raise ValueError("video must contain at least eight frames")
    indices = [
        (index * (frame_count - 1) * 2 + (sample_count - 1)) // (2 * (sample_count - 1))
        for index in range(sample_count)
    ]
    if len(set(indices)) != sample_count:
        raise ValueError("uniform frame selection produced duplicates")
    return indices


def _hash_array(hasher: Any, array: Any) -> None:
    hasher.update(str(tuple(array.shape)).encode())
    hasher.update(str(array.dtype).encode())
    hasher.update(array.tobytes(order="C"))


def _decode_audio(path: Path):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        streams = list(container.streams.audio)
        if len(streams) != 1:
            raise ValueError("exactly one audio stream is required")
        stream = streams[0]
        input_rate = int(stream.rate or stream.codec_context.sample_rate)
        if input_rate <= 0:
            raise ValueError("audio sample rate is unavailable")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        chunks = []
        for frame in container.decode(stream):
            if frame.sample_rate != input_rate:
                raise ValueError("sample rate changes within an audio file")
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
        if not chunks:
            raise ValueError("audio decoder returned no samples")
        waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise ValueError("audio waveform must be finite and nonzero")
    digest = hashlib.sha256()
    _hash_array(digest, waveform)
    return waveform, {
        "input_sample_rate": input_rate,
        "normalized_sample_rate": 24000,
        "decoded_samples_24khz": len(waveform),
        "normalized_content_sha256": digest.hexdigest(),
    }


def _decode_video(path: Path):
    import av
    import numpy as np

    frame_count = 0
    previous_pts: Fraction | None = None
    full_content = hashlib.sha256()
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("exactly one video stream is required")
        stream_index = streams[0].index
        for frame in container.decode(video=stream_index):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frames require presentation timestamps")
            pts = Fraction(frame.pts) * Fraction(frame.time_base)
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError("video PTS must be strictly increasing")
            previous_pts = pts
            rgb = frame.reformat(width=FRAME_SIZE[0], height=FRAME_SIZE[1], format="rgb24")
            pixels = rgb.to_ndarray()
            if pixels.shape != (*FRAME_SIZE, 3) or pixels.dtype != np.uint8:
                raise ValueError("decoded RGB frame shape/dtype differs")
            _hash_array(full_content, pixels)
            frame_count += 1
        if frame_count < NUM_VIDEO_FRAMES:
            raise ValueError("video has fewer than eight decoded frames")
        selected_indices = uniform_frame_indices(frame_count)

    selected_set = set(selected_indices)
    selected_frames = []
    with av.open(str(path)) as container:
        decoded_count = 0
        for frame in container.decode(video=0):
            if decoded_count in selected_set:
                selected_frames.append(
                    frame.reformat(
                        width=FRAME_SIZE[0], height=FRAME_SIZE[1], format="rgb24"
                    ).to_ndarray()
                )
            decoded_count += 1
    if decoded_count != frame_count or len(selected_frames) != NUM_VIDEO_FRAMES:
        raise ValueError("video changed between frame inventory and selection passes")
    frames = np.stack(selected_frames).astype(np.uint8, copy=False)
    return frames, {
        "decoded_frame_count": frame_count,
        "selected_frame_indices": selected_indices,
        "selected_pts_policy": "endpoint-inclusive uniform index selection",
        "frame_size": list(FRAME_SIZE),
        "normalized_content_sha256": full_content.hexdigest(),
    }


def prepare(media_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    acquisition = json.loads((media_dir / "result.json").read_text(encoding="utf-8"))
    members_path = media_dir / "members.json"
    if acquisition.get("status") != "PASS_MEDIA_ACQUISITION_RAW_HASH_DISJOINT":
        raise ValueError("complete raw media acquisition receipt required")
    if acquisition.get("completed") != 440 or acquisition.get("passed") != 440:
        raise ValueError("all 440 raw assets are required")
    members = json.loads(members_path.read_text(encoding="utf-8"))
    if len(members) != 440 or sha256_file(members_path) != acquisition["members_manifest_sha256"]:
        raise ValueError("raw media manifest identity mismatch")

    groups: dict[str, dict[str, Any]] = {}
    raw_owners: dict[tuple[str, str], str] = {}
    for item in members.values():
        if item.get("status") != "PASS":
            raise ValueError("only successful raw media receipts are allowed")
        path = media_dir / item["file"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError("raw media differs from its verified receipt")
        key = (item["modality"], item["sha256"])
        owner = raw_owners.setdefault(key, item["split"])
        if owner != item["split"]:
            raise ValueError("identical raw media hash crosses splits")
        group = groups.setdefault(item["source_group"], {"split": item["split"], "media": {}})
        if group["split"] != item["split"] or item["modality"] in group["media"]:
            raise ValueError("source group/split/media cardinality mismatch")
        group["media"][item["modality"]] = (path, item)
    if len(groups) != 220 or any(
        set(group["media"]) != {"audio", "video"} for group in groups.values()
    ):
        raise ValueError("exactly 220 paired source groups with both modalities are required")

    output_dir.mkdir(parents=True)
    (output_dir / "inputs").mkdir()
    start = time.monotonic()
    normalized_owners: dict[tuple[str, str], str] = {}
    rows = []
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "media_manifest_sha256": sha256_file(members_path),
        "groups": len(groups),
        "num_video_frames": NUM_VIDEO_FRAMES,
        "training": False,
        "evaluation": False,
        "sealed_audit_opened": False,
        "optimizer_updates": 0,
    }
    try:
        for index, (group_id, group) in enumerate(sorted(groups.items())):
            split = group["split"]
            audio_path, audio_receipt = group["media"]["audio"]
            video_path, video_receipt = group["media"]["video"]
            waveform, audio_info = _decode_audio(audio_path)
            frames, video_info = _decode_video(video_path)
            for modality, info in (("audio", audio_info), ("video", video_info)):
                key = (modality, info["normalized_content_sha256"])
                owner = normalized_owners.setdefault(key, split)
                if owner != split:
                    raise ValueError(f"normalized decoded {modality} overlap crosses splits")
            destination = output_dir / "inputs" / split / f"{group_id}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, audio=waveform[:, None], video=frames)
            row = {
                "source_group": group_id,
                "split": split,
                "audio_file": str(audio_path.relative_to(media_dir)),
                "video_file": str(video_path.relative_to(media_dir)),
                "audio_sha256": audio_receipt["sha256"],
                "video_sha256": video_receipt["sha256"],
                "input_file": str(destination.relative_to(output_dir)),
                "input_sha256": sha256_file(destination),
                "audio": audio_info,
                "video": video_info,
            }
            rows.append(row)
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_groups": index + 1,
                        "expected_groups": len(groups),
                        "elapsed_seconds": time.monotonic() - start,
                        "last_source_group": group_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if len({row["source_group"] for row in rows}) != len(rows):
            raise ValueError("duplicate source group in prepared output")
        (output_dir / "groups.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        receipt["status"] = "PASS_FULL_DECODE_AND_NORMALIZED_CONTENT_SPLIT_GATES"
    except Exception as error:
        receipt.update(status="FAILED", error=repr(error))
        raise
    finally:
        receipt.update(
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - start,
            completed_groups=len(rows),
            split_groups={
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "calibration", "validation")
            },
            normalized_content_hashes={
                modality: len({row[modality]["normalized_content_sha256"] for row in rows})
                for modality in ("audio", "video")
                if rows
            },
            arrays_npz_sha256={row["input_file"]: row["input_sha256"] for row in rows},
            artifact_hashes={
                file.name: sha256_file(file)
                for file in output_dir.iterdir()
                if file.is_file() and file.name != "result.json"
            },
        )
        (output_dir / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.media_dir, args.output_dir)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "completed_groups",
                    "split_groups",
                    "elapsed_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
