"""Decode fixed STARSS23 dev-train windows; no model fitting or evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import wave
from pathlib import Path

EVENTS = (
    "speech",
    "clapping",
    "telephone",
    "laughter",
    "domestic sounds",
    "footsteps",
    "door",
    "music",
    "instrument",
    "water tap",
    "bell",
    "knock",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_annotations(text):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError("expected six annotation columns")
        frame, label, source, azimuth, elevation, distance = map(int, fields)
        if frame < 0 or not 0 <= label <= 12 or source < 0:
            raise ValueError("invalid annotation identity")
        if not -180 <= azimuth <= 180 or not -90 <= elevation <= 90 or distance < 0:
            raise ValueError("invalid spatial annotation")
        rows.append((frame, 0 if label <= 1 else label - 1, source))
    return rows


def window_queries(rows, start_frame, end_frame):
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("invalid half-open window")
    counts = [
        len({f for f, c, _ in rows if c == label and start_frame <= f < end_frame})
        for label in range(len(EVENTS))
    ]
    return [
        {
            "event": event,
            "question": f"Is {event} annotated in this time interval?",
            "options": ["no", "yes"],
            "target": int(count > 0),
            "active_annotation_frames": count,
        }
        for event, count in zip(EVENTS, counts, strict=True)
    ]


def validate_pairs(report):
    if report.get("status") != "PASS" or report.get("record") != "doi:10.5281/zenodo.7880637":
        raise ValueError("unexpected source probe")
    pairs = report["pairs"]
    if len({p["clip"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate recording")
    if any(p["split"] != "train" for p in pairs):
        raise ValueError("probe permits dev-train only")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = validate_pairs(json.loads(args.source_result.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import av
    import numpy as np

    report = {
        "status": "RUNNING",
        "source_result_sha256": digest(args.source_result),
        "runner_sha256": digest(__file__),
        "windows": [],
        "annotation_tick_seconds": 0.1,
        "window_seconds": 5,
        "windows_per_recording": 2,
        "frames_per_window": 8,
        "video_selection": "first presentation timestamp at/after each bin midpoint",
        "audio_preprocessing": "all four original PCM16 channels, float32 / 32768",
        "task_scope": "annotated prominent-event presence; not exhaustive audible presence",
        "dev_test_accessed": False,
        "models_loaded": False,
    }
    try:
        for pair in pairs:
            for media in pair["files"].values():
                if digest(media["path"]) != media["sha256"]:
                    raise ValueError("source media hash mismatch")
            annotation = args.source_result.parent / f"{pair['clip']}.csv"
            if digest(annotation) != pair["annotation_sha256"]:
                raise ValueError("annotation hash mismatch")
            rows = parse_annotations(annotation.read_text(encoding="utf-8"))
            with wave.open(pair["files"]["audio"]["path"]) as wav:
                if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (4, 24000, 2):
                    raise ValueError("unexpected audio format")
                samples = np.frombuffer(wav.readframes(240000), dtype="<i2").reshape(-1, 4)
                if samples.shape != (240000, 4):
                    raise ValueError("recording shorter than fixed smoke windows")
                samples = samples.astype(np.float32) / 32768
            targets = np.array([start + (i + 0.5) * 5 / 8 for start in (0, 5) for i in range(8)])
            frames, times = [], []
            with av.open(pair["files"]["video"]["path"]) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("video frame has no timestamp")
                    timestamp = float(frame.pts * stream.time_base)
                    if timestamp >= targets[len(frames)]:
                        if timestamp - targets[len(frames)] > 0.05:
                            raise ValueError("video timestamp gap exceeds 50 ms")
                        frames.append(frame.to_ndarray(format="rgb24"))
                        times.append(timestamp)
                        if len(frames) == len(targets):
                            break
            if len(frames) != 16:
                raise ValueError("incomplete video windows")
            for index, start in enumerate((0, 5)):
                clip_id = f"{pair['clip']}:t{start:04d}-{start + 5:04d}"
                path = args.output_dir / f"{pair['clip']}-{start:04d}.npz"
                audio = samples[start * 24000 : (start + 5) * 24000]
                video = np.stack(frames[index * 8 : (index + 1) * 8])
                timestamps = np.array(times[index * 8 : (index + 1) * 8])
                if not np.isfinite(audio).all() or not (timestamps < start + 5).all():
                    raise ValueError("invalid decoded interval")
                np.savez_compressed(path, audio=audio, video=video, video_seconds=timestamps)
                with np.load(path, allow_pickle=False) as reload:
                    if not (
                        np.array_equal(audio, reload["audio"])
                        and np.array_equal(video, reload["video"])
                        and np.array_equal(timestamps, reload["video_seconds"])
                    ):
                        raise ValueError("window reload mismatch")
                report["windows"].append(
                    {
                        "id": clip_id,
                        "room": pair["room"],
                        "recording": pair["clip"],
                        "split": "dev-train",
                        "start_seconds": start,
                        "end_seconds": start + 5,
                        "audio_shape": list(audio.shape),
                        "video_shape": list(video.shape),
                        "video_seconds": timestamps.tolist(),
                        "path": str(path.resolve()),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                        "queries": window_queries(rows, start * 10, (start + 5) * 10),
                    }
                )
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "windows": len(report["windows"]),
                "queries": sum(len(w["queries"]) for w in report["windows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
