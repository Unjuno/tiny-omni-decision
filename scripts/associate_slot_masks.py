"""Fixed greedy mask-overlap association; only the initial box is available."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.diagnose_slot_masks import box_mask, iou


def associate(masks, frames, initial_box):
    masks = np.asarray(masks)
    if masks.ndim != 4 or not np.isfinite(masks).all() or min(masks.shape) < 1:
        raise ValueError("invalid masks")
    if (
        len(frames) != len(masks)
        or frames[0] != 0
        or any(b <= a for a, b in zip(frames, frames[1:], strict=False))
    ):
        raise ValueError("frames must start at zero and strictly increase")
    hard = masks.argmax(1)
    region = box_mask(initial_box, *hard.shape[-2:])
    selected = int(np.argmax([iou(hard[0] == s, region) for s in range(masks.shape[1])]))
    ids = [selected]
    previous = hard[0] == selected
    for frame in hard[1:]:
        scores = [iou(frame == s, previous) for s in range(masks.shape[1])]
        best = max(scores)
        if scores[selected] != best:
            selected = int(np.argmax(scores))
        ids.append(selected)
        previous = frame == selected
    return ids


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
    with zipfile.ZipFile(args.annotations) as archive:
        annotations = json.loads(archive.read("all_train.json"))
    rows = []
    for original in result["rows"]:
        video = original["video_id"]
        path = args.run / (video + ".npz")
        if sha(path) != result["artifacts"][path.name]:
            raise ValueError("mask changed")
        tracks = [t for t in annotations[video]["object_tracking"] if t["is_occluder"]]
        if len(tracks) != 1:
            raise ValueError("ambiguous track")
        track = tracks[0]
        boxes = {
            int(i): b
            for i, b, masked in zip(
                track["frame_ids"], track["bounding_boxes"], track["is_masked"], strict=True
            )
            if not masked
        }
        with np.load(path, allow_pickle=False) as cache:
            masks = cache["masks"]
            frames = cache["frame_ids"].tolist()
        ids = associate(masks, frames, boxes[0])
        hard = masks.argmax(1)
        if frames != [p["frame"] for p in original["points"]]:
            raise ValueError("evaluation frame mismatch")
        points = []
        for index, (frame, slot, old) in enumerate(
            zip(frames, ids, original["points"], strict=True)
        ):
            score = iou(hard[index] == slot, box_mask(boxes[frame], *hard.shape[-2:]))
            points.append(
                dict(
                    frame=frame,
                    slot=slot,
                    associated_iou=score,
                    fixed_iou=old["fixed_iou"],
                    oracle_iou=old["oracle_iou"],
                )
            )
        rows.append(dict(video_id=video, points=points))
    points = [p for r in rows for p in r["points"] if p["frame"] > 0]
    report = dict(
        policy=(
            "greedy previous hard-mask IoU; ties keep previous index then lowest index; "
            "initial box only"
        ),
        scope="sparse saved annotation frames; oracle initialized; not Decision accuracy",
        input_sha256=sha(args.run / "result.json"),
        annotation_sha256=sha(args.annotations),
        script_sha256=sha(Path(__file__)),
        later_points=len(points),
        micro_mean={
            k: float(np.mean([p[k] for p in points]))
            for k in ["associated_iou", "fixed_iou", "oracle_iou"]
        },
        rows=rows,
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
