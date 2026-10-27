"""Check exact audio coverage and in-range video PTS on frozen train samples."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av

import tiny_omni_decision.audio_interval as interval_module
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.audio_interval import sample_interval

SAMPLE_SHA = "9d7df2419a4b3b63b23d1da94f571cf2ba29f3492b433caad0d530c756531126"
PROTOCOL = Path("docs/pretrained_reuse/R2_SOURCE_REVIEW_CLOCK_PROTOCOL_20261027.md")


def first_video_frame(path, start_us, end_us):
    """Return first decoded frame PTS inside the half-open event interval."""
    start, end = Fraction(start_us, 1_000_000), Fraction(end_us, 1_000_000)
    if start < 0 or start >= end:
        raise ValueError("invalid event interval")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = Fraction(stream.time_base)
        container.seek(
            max(0, int(start / time_base)), stream=stream, backward=True, any_frame=False
        )
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frame lacks timestamp")
            t = Fraction(frame.pts) * Fraction(frame.time_base)
            if t >= end:
                break
            if t >= start:
                return {"pts": frame.pts, "time_base": str(frame.time_base), "seconds": float(t)}
    raise ValueError("no video frame inside event")


def audio_coverage(path, start_us, end_us):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        blocks = [(frame.pts, frame.time_base, frame.samples) for frame in container.decode(stream)]
    begin, end = sample_interval(blocks, sample_rate, start_us, end_us)
    return {
        "sample_rate": sample_rate,
        "decoded_blocks": len(blocks),
        "first_pts": blocks[0][0],
        "first_time_base": str(blocks[0][1]),
        "event_sample_start": begin,
        "event_sample_end": end,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        Path(interval_module.__file__).resolve()
        != Path("src/tiny_omni_decision/audio_interval.py").resolve()
    ):
        raise ValueError("wrong audio interval module; set PYTHONPATH=src;.")
    if args.output.exists() or digest(args.sample) != SAMPLE_SHA:
        raise ValueError("output exists or frozen sample hash mismatch")
    sample = json.loads(args.sample.read_bytes())
    if len(sample["samples"]) != 16:
        raise ValueError("unexpected sample count")
    rows = []
    for row in sample["samples"]:
        path = Path(row["media_path"])
        if digest(path) != row["media_sha256"]:
            raise ValueError("media hash mismatch")
        start, end = row["audio_interval_us"]
        result = {
            "record_id": row["record_id"],
            "category": row["category"],
            "media_sha256": row["media_sha256"],
            "interval_us": [start, end],
        }
        for name, function in (("audio", audio_coverage), ("video", first_video_frame)):
            try:
                result[name] = {"status": "PASS", **function(path, start, end)}
            except (ValueError, IndexError, av.error.FFmpegError) as error:
                result[name] = {"status": "HELD", "reason": str(error)}
        rows.append(result)
        print(row["record_id"], result["audio"]["status"], result["video"]["status"], flush=True)
    result = {
        "sample_sha256": SAMPLE_SHA,
        "script_sha256": digest(Path(__file__)),
        "audio_interval_sha256": digest(Path("src/tiny_omni_decision/audio_interval.py")),
        "protocol_sha256_lf": hashlib.sha256(
            PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "pyav_version": av.__version__,
        "records": rows,
        "scope": "container PTS and decoded samples only; sync and semantic correctness unproven",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print("result SHA-256", digest(args.output))


if __name__ == "__main__":
    main()
