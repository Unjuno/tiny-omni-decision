"""Recompute train-only oracle slot diagnostics from hashed saved masks."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.diagnose_slot_masks import compare_masks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    protocol = json.loads((args.run / "protocol.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    if sha(args.annotations) != protocol["annotation_sha256"]:
        raise ValueError("annotations changed")
    if sha(Path(__file__).with_name("diagnose_slot_masks.py")) != protocol["helper_sha256"]:
        raise ValueError("metric implementation changed")
    with zipfile.ZipFile(args.annotations) as archive:
        annotations = json.loads(archive.read("all_train.json"))
    if [r["video_id"] for r in result["rows"]] != protocol["video_ids"]:
        raise ValueError("video selection changed")
    per_video = []
    for row in result["rows"]:
        path = args.run / (row["video_id"] + ".npz")
        if sha(path) != result["artifacts"][path.name]:
            raise ValueError("mask cache changed")
        tracks = [t for t in annotations[row["video_id"]]["object_tracking"] if t["is_occluder"]]
        if len(tracks) != 1:
            raise ValueError("ambiguous object")
        track = tracks[0]
        boxes = {
            int(i): b
            for i, b, masked in zip(
                track["frame_ids"], track["bounding_boxes"], track["is_masked"], strict=True
            )
            if not masked
        }
        with np.load(path, allow_pickle=False) as cache:
            replay = compare_masks(cache["masks"], cache["frame_ids"].tolist(), boxes)
        if any(row.get(k) != v for k, v in replay.items()):
            raise ValueError("exact replay failed")
        points = [p for p in replay.get("points", []) if p["frame"] > 0]
        per_video.append(
            dict(
                video_id=row["video_id"],
                points=len(points),
                **{
                    k: float(np.mean([p[k] for p in points])) if points else None
                    for k in ("fixed_iou", "oracle_iou", "static_iou", "largest_slot_area")
                },
            )
        )
    report = dict(
        exact_replay=True,
        protocol_sha256=sha(args.run / "protocol.json"),
        result_sha256=sha(args.run / "result.json"),
        per_video=per_video,
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
