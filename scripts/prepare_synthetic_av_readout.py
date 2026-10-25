"""Render deterministic audio/video assets for the synthetic AV readout task."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw

from tiny_omni_decision.synthetic_av_task import SEQUENCE_LENGTH, build_manifest, validate_manifest

FRAME_SIZE = 224
FRAME_RATE = 2
SAMPLE_RATE_HZ = 32_000
EVENT_SECONDS = 0.5
TONE_FREQUENCIES_HZ = (440.0, 554.365, 659.255, 783.991)
COLORS = ((235, 73, 62), (39, 169, 116), (65, 119, 232), (241, 184, 49))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame(symbol: int) -> np.ndarray:
    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (16, 19, 26))
    draw = ImageDraw.Draw(image)
    box = (48, 48, 176, 176)
    fill = COLORS[symbol]
    if symbol == 0:
        draw.ellipse(box, fill=fill, outline=(255, 255, 255), width=5)
    elif symbol == 1:
        draw.rectangle(box, fill=fill, outline=(255, 255, 255), width=5)
    elif symbol == 2:
        draw.polygon(((112, 42), (183, 176), (41, 176)), fill=fill, outline=(255, 255, 255))
    elif symbol == 3:
        draw.polygon(
            ((112, 38), (186, 112), (112, 186), (38, 112)),
            fill=fill,
            outline=(255, 255, 255),
        )
    else:
        raise ValueError(f"unsupported symbol: {symbol}")
    return np.asarray(image, dtype=np.uint8)


def _write_video(path: Path, symbols: list[int]) -> list[Path]:
    frame_dir = path.parent.parent / "frames" / path.stem
    frame_dir.mkdir(parents=True, exist_ok=False)
    frame_paths = []
    frames = []
    for index, symbol in enumerate(symbols):
        array = _frame(symbol)
        frame_path = frame_dir / f"frame-{index:02d}.png"
        Image.fromarray(array).save(frame_path, format="PNG", compress_level=9, optimize=False)
        frame_paths.append(frame_path)
        frames.append(array)

    with av.open(str(path), mode="w", format="avi") as container:
        stream = container.add_stream("ffv1", rate=FRAME_RATE)
        stream.width = FRAME_SIZE
        stream.height = FRAME_SIZE
        stream.pix_fmt = "bgr0"
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return frame_paths


def _write_audio(path: Path, symbols: list[int]) -> list[dict[str, Any]]:
    samples_per_event = round(SAMPLE_RATE_HZ * EVENT_SECONDS)
    fade_samples = round(SAMPLE_RATE_HZ * 0.004)
    segments = []
    waveform = np.empty(samples_per_event * len(symbols), dtype=np.int16)
    for index, symbol in enumerate(symbols):
        start = index * samples_per_event
        stop = start + samples_per_event
        time = np.arange(samples_per_event, dtype=np.float64) / SAMPLE_RATE_HZ
        signal = 0.35 * np.sin(2.0 * np.pi * TONE_FREQUENCIES_HZ[symbol] * time)
        envelope = np.ones(samples_per_event, dtype=np.float64)
        envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
        waveform[start:stop] = np.round(signal * envelope * 32767).astype(np.int16)
        segments.append(
            {
                "event_index": index,
                "symbol_id": symbol,
                "tone_frequency_hz": TONE_FREQUENCIES_HZ[symbol],
                "start_seconds": index * EVENT_SECONDS,
                "end_seconds": (index + 1) * EVENT_SECONDS,
                "sample_start": start,
                "sample_end": stop,
            }
        )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE_HZ)
        output.writeframes(waveform.astype("<i2", copy=False).tobytes())
    return segments


def render_assets(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Render one lossless video and WAV per unique sequence, refusing overwrite."""
    validate_manifest(manifest)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    sequences: dict[str, list[int]] = {}
    for row in manifest["examples"]:
        sequences[row["audio_sequence_id"]] = row["audio_symbols"]
        sequences[row["video_sequence_id"]] = row["video_symbols"]
    output_dir.mkdir(parents=True, exist_ok=False)
    videos: dict[str, dict[str, Any]] = {}
    audio: dict[str, dict[str, Any]] = {}
    for sequence_id, symbols in sorted(sequences.items()):
        video_path = output_dir / "videos" / f"{sequence_id}.avi"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_paths = _write_video(video_path, symbols)
        videos[sequence_id] = {
            "path": video_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(video_path),
            "frame_count": SEQUENCE_LENGTH,
            "frame_paths": [path.relative_to(output_dir).as_posix() for path in frame_paths],
            "frame_sha256": [_sha256(path) for path in frame_paths],
            "width": FRAME_SIZE,
            "height": FRAME_SIZE,
            "frame_rate_hz": FRAME_RATE,
            "symbols": symbols,
        }
        audio_path = output_dir / "audio" / f"{sequence_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        segments = _write_audio(audio_path, symbols)
        audio[sequence_id] = {
            "path": audio_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(audio_path),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "sample_width_bytes": 2,
            "sample_count": len(symbols) * round(SAMPLE_RATE_HZ * EVENT_SECONDS),
            "duration_seconds": len(symbols) * EVENT_SECONDS,
            "segments": segments,
            "symbols": symbols,
        }

    rendered = {**manifest, "assets": {"videos": videos, "audio": audio}}
    for row in rendered["examples"]:
        row["video_path"] = videos[row["video_sequence_id"]]["path"]
        row["video_sha256"] = videos[row["video_sequence_id"]]["sha256"]
        row["audio_path"] = audio[row["audio_sequence_id"]]["path"]
        row["audio_sha256"] = audio[row["audio_sequence_id"]]["sha256"]
    (output_dir / "render-manifest.json").write_text(
        json.dumps(rendered, indent=2) + "\n", encoding="utf-8"
    )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-groups", type=int, required=True)
    parser.add_argument("--validation-groups", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render_assets(
        build_manifest(args.train_groups, args.validation_groups, args.seed), args.output_dir
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "groups": len(result["groups"]),
                "examples": len(result["examples"]),
                "unique_sequences": len(result["assets"]["videos"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
