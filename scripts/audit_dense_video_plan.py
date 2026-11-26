"""Decode PTS only for all acquired train occlusion examples; no model loading."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import av

from tiny_omni_decision.temporal_sampling_plan import plan_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    archive = root / "perception-test-source-audit-01/train_annotations.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != (
        "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"
    ):
        raise ValueError("annotation changed")
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read("all_train.json"))
    source = root / "perception-correspondence-development-windows-01/result.json"
    windows = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "perception-correspondence-train-media-01/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    media = {r["video_id"]: r for r in manifest["clips"]}
    results = []
    for row in windows["windows"]:
        if row["split"] != "train" or not any(
            "occlusion game" in q["question"] for q in annotations[row["id"]]["mc_question"]
        ):
            continue
        path = Path(media[row["id"]]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["media_sha256"]:
            raise ValueError("media changed")
        times = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise ValueError("missing PTS")
                times.append(float(frame.pts * frame.time_base))
        start, end = [t / 1e6 for t in row["interval_us"]]
        plan = plan_windows(times, start=start, end=end)
        old = [times[i] for i in row["selected_frame_indices"]]
        results.append(
            dict(
                id=row["id"],
                cut_frame=row["cut_frame"],
                old_max_gap=max(b - a for a, b in zip(old, old[1:], strict=False)),
                **plan,
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                fps=4,
                width=8,
                scope="train input feasibility, no inference",
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                videos=results,
                windows=sum(len(r["windows"]) for r in results),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(dict(videos=len(results), windows=sum(len(r["windows"]) for r in results))))


if __name__ == "__main__":
    main()
