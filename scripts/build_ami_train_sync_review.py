"""Build a train-only audio-envelope/video timing review from frozen AMI rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

FRAME_COUNT = 12
RMS_BIN_COUNT = 24
ROWS_PER_PAGE = 12
CELL_WIDTH = 118
CELL_HEIGHT = 128
LABEL_WIDTH = 190
ROW_HEIGHT = 154


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_times(start_s: float, end_s: float, count: int = FRAME_COUNT) -> list[float]:
    if not np.isfinite([start_s, end_s]).all() or start_s < 0 or end_s <= start_s:
        raise ValueError("sample interval must be finite, nonnegative, and nonempty")
    if count < 2:
        raise ValueError("at least two frame samples are required")
    return np.linspace(start_s, end_s, count).astype(float).tolist()


def wav_rms_bins(
    path: Path, start_s: float, end_s: float, count: int = RMS_BIN_COUNT
) -> list[float]:
    if count < 1 or start_s < 0 or end_s <= start_s:
        raise ValueError("invalid RMS interval")
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit PCM WAV")
        rate = audio.getframerate()
        if rate != 16000:
            raise ValueError(f"expected 16 kHz WAV, got {rate}")
        first = max(0, int(start_s * rate))
        last = min(audio.getnframes(), int(end_s * rate))
        if last <= first:
            raise ValueError("audio interval is outside the WAV")
        audio.setpos(first)
        signal = np.frombuffer(audio.readframes(last - first), dtype="<i2").astype(np.float64)
    edges = np.linspace(0, signal.size, count + 1, dtype=int)
    rms = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        values = signal[left:right]
        rms.append(float(np.sqrt(np.mean(values * values))) if values.size else 0.0)
    peak = float(np.percentile(rms, 95)) if rms else 0.0
    return [float(value / peak) if peak > 0 else 0.0 for value in rms]


def _frame_at(path: Path, timestamp_s: float) -> tuple[Image.Image, float]:
    container = av.open(str(path))
    try:
        stream = next(item for item in container.streams if item.type == "video")
        container.seek(
            max(0, int(timestamp_s / float(stream.time_base))),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual_s = float(frame.pts * stream.time_base)
            selected = frame
            if actual_s >= timestamp_s:
                break
        if selected is None:
            raise ValueError(f"no frame found near {timestamp_s:.3f}s in {path.name}")
        return selected.to_image().convert("RGB"), float(selected.pts * stream.time_base)
    finally:
        container.close()


def _render_page(samples: list[dict[str, Any]], output_path: Path, page_index: int) -> None:
    font = ImageFont.load_default()
    width = LABEL_WIDTH + FRAME_COUNT * CELL_WIDTH
    height = 34 + len(samples) * ROW_HEIGHT
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (LABEL_WIDTH + 4, 8),
        f"Train-only AV timing screen — page {page_index}",
        fill="black",
        font=font,
    )
    for row_index, sample in enumerate(samples):
        top = 34 + row_index * ROW_HEIGHT
        draw.text(
            (4, top + 18),
            f"{sample['sample_id']}\n{sample['start_s']:.2f}–{sample['end_s']:.2f}s\n{sample['target_camera']}",
            fill="black",
            font=font,
        )
        for frame_index, (frame, timestamp) in enumerate(sample.pop("_frames")):
            frame.thumbnail((CELL_WIDTH - 6, CELL_HEIGHT - 16))
            x = LABEL_WIDTH + frame_index * CELL_WIDTH + (CELL_WIDTH - frame.width) // 2
            y = top + 12
            sheet.paste(frame, (x, y))
            draw.text(
                (LABEL_WIDTH + frame_index * CELL_WIDTH + 2, top + 1),
                f"{timestamp:.2f}",
                fill="black",
                font=font,
            )
            draw.rectangle(
                (
                    LABEL_WIDTH + frame_index * CELL_WIDTH,
                    top + 2,
                    LABEL_WIDTH + (frame_index + 1) * CELL_WIDTH - 1,
                    top + CELL_HEIGHT,
                ),
                outline="#aaa",
            )
        bar_top = top + CELL_HEIGHT + 5
        values = sample["audio_rms_normalized"]
        max_height = ROW_HEIGHT - CELL_HEIGHT - 14
        for index, value in enumerate(values):
            x0 = LABEL_WIDTH + int(index * FRAME_COUNT * CELL_WIDTH / len(values))
            x1 = LABEL_WIDTH + max(
                int((index + 1) * FRAME_COUNT * CELL_WIDTH / len(values)), x0 + 1
            )
            bar_height = int(max_height * max(0.0, min(1.0, value)))
            draw.rectangle(
                (x0, bar_top + max_height - bar_height, x1, bar_top + max_height), fill="#3976bd"
            )
        draw.text(
            (LABEL_WIDTH + 4, bar_top + max_height),
            "mixed-headset WAV RMS (normalized per sample)",
            fill="#345",
            font=font,
        )
    sheet.save(output_path)


def build_review(
    sample_manifest_path: Path,
    identity_audit_path: Path,
    media_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    identity_audit = json.loads(identity_audit_path.read_text(encoding="utf-8"))
    samples = sample_manifest.get("samples", [])
    if sample_manifest.get("sample_count") != len(samples):
        raise ValueError("sample manifest count mismatch")
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("duplicate sample IDs")
    if identity_audit.get("cross_split_exact_duplicate_group_count") != 0:
        raise ValueError("selected media identity audit reports cross-split exact duplicates")
    media_records = {
        (entry["meeting"], entry["file"]): entry for entry in identity_audit.get("files", [])
    }
    inputs = {
        "sample_manifest": sha256_file(sample_manifest_path),
        "identity_audit": sha256_file(identity_audit_path),
    }
    rendered_samples = []
    verified_media_hashes: dict[Path, str] = {}
    for sample in samples:
        meeting = sample["meeting"]
        video_name = f"{meeting}.{sample['target_camera']}.avi"
        audio_name = f"{meeting}.Mix-Headset.wav"
        video_record = media_records.get((meeting, video_name))
        audio_record = media_records.get((meeting, audio_name))
        if not video_record or not audio_record:
            raise ValueError(f"missing SA media receipt for {sample['sample_id']}")
        if video_record["split"] != "train" or audio_record["split"] != "train":
            raise ValueError(f"non-train media in sample {sample['sample_id']}")
        video_path = media_root / meeting / "video" / video_name
        audio_path = media_root / meeting / "audio" / audio_name
        if video_path not in verified_media_hashes:
            verified_media_hashes[video_path] = sha256_file(video_path)
        if audio_path not in verified_media_hashes:
            verified_media_hashes[audio_path] = sha256_file(audio_path)
        if verified_media_hashes[video_path] != video_record["sha256"]:
            raise ValueError(f"video hash mismatch for {sample['sample_id']}")
        if verified_media_hashes[audio_path] != audio_record["sha256"]:
            raise ValueError(f"audio hash mismatch for {sample['sample_id']}")
        times = sample_times(float(sample["start_s"]), float(sample["end_s"]))
        frames = []
        frame_records = []
        for requested_s in times:
            image, actual_s = _frame_at(video_path, requested_s)
            image.thumbnail((CELL_WIDTH - 6, CELL_HEIGHT - 16))
            frames.append((image, actual_s))
            frame_records.append(
                {
                    "requested_s": round(requested_s, 6),
                    "actual_pts_s": round(actual_s, 6),
                    "delta_s": round(actual_s - requested_s, 6),
                }
            )
        rendered_samples.append(
            {
                "sample_id": sample["sample_id"],
                "meeting": meeting,
                "agent": sample["agent"],
                "target_camera": sample["target_camera"],
                "start_s": sample["start_s"],
                "end_s": sample["end_s"],
                "audio_rms_normalized": wav_rms_bins(
                    audio_path, sample["start_s"], sample["end_s"]
                ),
                "audio_sha256": audio_record["sha256"],
                "video_sha256": video_record["sha256"],
                "frame_samples": frame_records,
                "_frames": frames,
            }
        )

    output_dir.mkdir(parents=True)
    pages = []
    for start in range(0, len(rendered_samples), ROWS_PER_PAGE):
        page_no = start // ROWS_PER_PAGE + 1
        path = output_dir / f"train-sync-contact-sheet-{page_no:02d}.png"
        page_samples = rendered_samples[start : start + ROWS_PER_PAGE]
        _render_page(page_samples, path, page_no)
        pages.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sample_ids": [sample["sample_id"] for sample in page_samples],
            }
        )
    result = {
        "schema_version": 1,
        "purpose": "train-only AV timing preflight; no model inference or selection",
        "review_status": "NOT_ADJUDICATED",
        "review_status_note": "Mixed-headset RMS cannot isolate a speaker or prove lip-sync.",
        "inputs": inputs,
        "sample_count": len(rendered_samples),
        "frame_count_per_sample": FRAME_COUNT,
        "audio_rms_bin_count": RMS_BIN_COUNT,
        "audio_rms_note": "Normalized mixed-headset RMS; not speaker diarization.",
        "environment": {"pyav": av.__version__, "numpy": np.__version__, "pillow": PIL.__version__},
        "pages": pages,
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key not in {"audio_rms_normalized", "_frames"}
            }
            for sample in rendered_samples
        ],
        "limitations": [
            "RMS is an activity proxy; it does not isolate the annotated speaker.",
            "Visual inspection of sampled frames cannot establish identity or precise lip-sync.",
            "Only the frozen SA seed-17 review sample is included; SB/SC media are excluded.",
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--identity-audit", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_review(
        args.sample_manifest, args.identity_audit, args.media_root, args.output_dir
    )
    print(json.dumps({"sample_count": result["sample_count"], "pages": result["pages"]}, indent=2))


if __name__ == "__main__":
    main()
