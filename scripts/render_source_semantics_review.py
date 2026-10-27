"""Render three event-time video frames for four frozen train-source samples."""

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

from scripts.train_source_unimodal_cv import digest

SAMPLE_SHA = "9d7df2419a4b3b63b23d1da94f571cf2ba29f3492b433caad0d530c756531126"


def sample_times(interval_us):
    if len(interval_us) != 2 or interval_us[0] < 0 or interval_us[1] <= interval_us[0]:
        raise ValueError("invalid event interval")
    start, end = (x / 1e6 for x in interval_us)
    return [start + (end - start) * fraction for fraction in (0.1, 0.5, 0.9)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.sample) != SAMPLE_SHA:
        raise ValueError("output exists or sample hash mismatch")
    sample = json.loads(args.sample.read_bytes())
    rows = [r for r in sample["samples"] if r["category"] == "none_related_candidate"]
    if len(rows) != 4:
        raise ValueError("unexpected sample count")
    width, height = 384, 216
    canvas = Image.new("RGB", (3 * width, len(rows) * (height + 25)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        path = Path(row["media_path"])
        if digest(path) != row["media_sha256"]:
            raise ValueError("media hash mismatch")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for j, time in enumerate(sample_times(row["audio_interval_us"])):
                container.seek(
                    max(0, int(time / float(stream.time_base))),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
                selected = None
                for frame in container.decode(stream):
                    if frame.pts is not None and float(frame.pts * frame.time_base) >= time:
                        selected = frame
                        break
                if selected is None:
                    raise ValueError("frame not found")
                actual = float(selected.pts * selected.time_base)
                if actual > row["audio_interval_us"][1] / 1e6:
                    raise ValueError("frame outside event interval")
                canvas.paste(
                    selected.to_image().resize((width, height)), (j * width, i * (height + 25))
                )
                draw.text(
                    (j * width + 2, i * (height + 25) + height + 3),
                    f"{row['record_id']} t={actual:.3f}s",
                    fill="black",
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        canvas.save(out, format="PNG")
    print("still frame montage SHA-256:", digest(args.output))


if __name__ == "__main__":
    main()
