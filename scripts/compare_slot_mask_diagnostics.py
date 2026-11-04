"""Compare oracle slot-mask diagnostics only on identical videos/frame geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate):
    a, b = reference["rows"], candidate["rows"]
    ids = [r["video_id"] for r in a]
    if not ids or len(set(ids)) != len(ids) or ids != [r["video_id"] for r in b]:
        raise ValueError("video identities/order differ or duplicate")
    keys = ("fixed_iou", "oracle_iou", "static_iou")
    rows = []
    total = 0
    for old, new in zip(a, b, strict=True):
        x, y = old.get("points", []), new.get("points", [])
        frames = [p["frame"] for p in x]
        if len(set(frames)) != len(frames) or frames != [p["frame"] for p in y]:
            raise ValueError("frame identities/order differ or duplicate")
        for p, q in zip(x, y, strict=True):
            if p["static_iou"] != q["static_iou"]:
                raise ValueError("static geometry differs")
            values = [r[k] for r in (p, q) for k in keys]
            if not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
                raise ValueError("invalid IoU")
        later = [(p, q) for p, q in zip(x, y, strict=True) if p["frame"] > 0]
        if not later:
            raise ValueError("no later frame observations")
        total += len(later)
        rows.append(
            dict(
                video_id=old["video_id"],
                points=len(later),
                delta={k: float(np.mean([q[k] - p[k] for p, q in later])) for k in keys},
            )
        )
    return dict(
        videos=len(rows),
        later_points=total,
        frame_alignment_exact=True,
        static_geometry_exact=True,
        per_video=rows,
        macro_video_delta={k: float(np.mean([r["delta"][k] for r in rows])) for k in keys},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    report = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    report["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [("reference", args.reference), ("candidate", args.candidate)]
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
