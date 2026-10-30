"""Create all complete fixed five-second windows from verified STARSS23 media."""

from __future__ import annotations

import argparse
import json
import math
import wave
import zipfile
from collections import Counter
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.fetch_starss23_frozen_media import load_splits, write_json
from scripts.fetch_starss23_smoke import PINNED
from scripts.probe_starss23_windows import parse_annotations, window_queries
from scripts.verify_starss23_acquisition import validate_acquisition


def intervals(audio_seconds, video_seconds):
    if not all(math.isfinite(t) and t > 0 for t in (audio_seconds, video_seconds)):
        raise ValueError("invalid media duration")
    # No padding or label-dependent selection. Omit only an incomplete final interval.
    return [(5 * i, 5 * (i + 1)) for i in range(math.floor(min(audio_seconds, video_seconds) / 5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("progress", "splits", "summary", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.progress.read_text())
    if report.get("summary_sha256") != sha256_file(args.summary):
        raise ValueError("split summary changed")
    if sha256_file(args.metadata) != PINNED["metadata_dev.zip"]:
        raise ValueError("metadata changed")
    rows = load_splits(args.splits, json.loads(args.summary.read_text()))
    verified = validate_acquisition(report, rows)
    import av

    args.output_dir.mkdir(parents=True, exist_ok=False)
    paired = {row["clip"]: row for row in report["pairs"]}
    counts = Counter()
    positives = Counter()
    result = {
        "status": "RUNNING",
        "verified_acquisition": verified,
        "acquisition_sha256": sha256_file(args.progress),
        "summary_sha256": sha256_file(args.summary),
        "metadata_sha256": sha256_file(args.metadata),
        "runner_sha256": sha256_file(Path(__file__)),
        "window_seconds": 5,
        "selection": "all complete nonoverlapping intervals",
        "official_dev_test_accessed": False,
        "dropped_partial_seconds": {},
    }
    try:
        with zipfile.ZipFile(args.metadata) as metadata:
            for split in ("train", "calibration", "validation"):
                with (args.output_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as out:
                    for row in rows:
                        if row["split"] != split:
                            continue
                        media = paired[row["clip"]]["files"]
                        with wave.open(media["audio"]["path"]) as audio:
                            if (
                                audio.getnchannels(),
                                audio.getframerate(),
                                audio.getsampwidth(),
                            ) != (4, 24000, 2):
                                raise ValueError("unexpected source audio format")
                            audio_seconds = audio.getnframes() / audio.getframerate()
                        with av.open(media["video"]["path"]) as video:
                            stream = video.streams.video[0]
                            if stream.duration is None or stream.start_time not in (None, 0):
                                raise ValueError("unsupported video timeline")
                            video_seconds = float(stream.duration * stream.time_base)
                        windows = intervals(audio_seconds, video_seconds)
                        if not windows:
                            raise ValueError("recording contains no complete window")
                        annotations = parse_annotations(
                            metadata.read(row["member"]).decode("utf-8")
                        )
                        result["dropped_partial_seconds"][row["clip"]] = (
                            min(audio_seconds, video_seconds) - windows[-1][1]
                        )
                        for start, end in windows:
                            queries = window_queries(annotations, start * 10, end * 10)
                            entry = {
                                "id": f"{row['clip']}:t{start:04d}-{end:04d}",
                                "recording": row["clip"],
                                "room": row["room"],
                                "split": split,
                                "source_revision": row["source_revision"],
                                "start_seconds": start,
                                "end_seconds": end,
                                "media": media,
                                "queries": queries,
                            }
                            out.write(json.dumps(entry, sort_keys=True) + "\n")
                            counts[split] += 1
                            if split == "train":
                                positives.update(q["event"] for q in queries if q["target"])
        result["splits"] = {
            split: {
                "windows": counts[split],
                "queries": counts[split] * 12,
                "sha256": sha256_file(args.output_dir / f"{split}.jsonl"),
            }
            for split in ("train", "calibration", "validation")
        }
        result["train_positive_queries"] = dict(positives)
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
