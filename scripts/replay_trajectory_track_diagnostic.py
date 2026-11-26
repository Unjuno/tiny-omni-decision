"""Replay exact-frame train tracking diagnostic from immutable cached features."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def compare(geometry, frame_ids, boxes):
    if 0 not in boxes:
        return dict(status="NO_FRAME0_BOX")
    box = boxes[0]
    xy = geometry[:, 0, :2]
    ids = np.flatnonzero(((xy >= box[:2]) & (xy <= box[2:])).all(-1))
    if not len(ids):
        return dict(status="NO_INITIAL_GRID_POINT")
    center = (box[:2] + box[2:]) / 2
    query = int(ids[np.argmin(np.linalg.norm(xy[ids] - center, axis=-1))])
    points = []
    for index, frame in enumerate(frame_ids):
        if frame == 0 or frame not in boxes:
            continue
        box = boxes[frame]
        center = (box[:2] + box[2:]) / 2
        predicted = geometry[query, index, :2]
        points.append(
            dict(
                frame=frame,
                error=float(np.linalg.norm(predicted - center)),
                static_error=float(np.linalg.norm(xy[query] - center)),
                inside=bool(((predicted >= box[:2]) & (predicted <= box[2:])).all()),
            )
        )
    return dict(status="EXACT_FRAME_DIAGNOSTIC", query_index=query, points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    archive = ROOT / "perception-test-source-audit-01/train_annotations.zip"
    if sha(archive) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("annotations changed")
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read("all_train.json"))
    features = ROOT / "perception-occlusion-trajectory-features-01/result.json"
    old_path = ROOT / "perception-occlusion-grid-track-exact-train-01.json"
    old = json.loads(old_path.read_text())
    if sha(features) != old["features_sha256"]:
        raise ValueError("feature ledger changed")
    rows = []
    for row in json.loads(features.read_text())["features"]:
        if row["split"] != "train":
            continue
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise ValueError("feature changed")
        geometry = np.load(path, allow_pickle=False)[:, :128].reshape(256, 32, 4)
        tracks = [t for t in annotations[row["video_id"]]["object_tracking"] if t["is_occluder"]]
        if len(tracks) != 1:
            raise ValueError("ambiguous marked track")
        track = tracks[0]
        boxes = {
            int(i): np.asarray(b)
            for i, b, m in zip(
                track["frame_ids"], track["bounding_boxes"], track["is_masked"], strict=True
            )
            if not m
        }
        rows.append(
            dict(video_id=row["video_id"], **compare(geometry, row["frame_indices"], boxes))
        )
    if rows != old["rows"]:
        raise ValueError("original diagnostic reproduction failed")
    differences = np.array(
        [
            np.mean([p["error"] - p["static_error"] for p in row["points"]])
            for row in rows
            if row.get("points")
        ]
    )
    generator = np.random.default_rng(17)
    means = [
        generator.choice(differences, len(differences), replace=True).mean() for _ in range(10000)
    ]
    write(
        args.output,
        dict(
            original_sha256=sha(old_path),
            exact_reproduction=True,
            clusters=len(differences),
            draws=10000,
            seed=17,
            method="equal-video means, replacement bootstrap, percentile95",
            mean_difference=float(differences.mean()),
            interval=np.percentile(means, [2.5, 97.5]).tolist(),
        ),
    )
    print("PASS_EXACT_REPRODUCTION")


if __name__ == "__main__":
    main()
