"""Audit pinned Touch and Go annotations and one recording's timestamp arrays.

No video/GelSight payloads, models, training, split creation or quality scores.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BASE = Path("C:/CodexArtifacts/pretrained-reuse-r2/touch-and-go-source-audit-20261027-attempt-01")
PINS = {
    "label.txt": "36f358e2f0ec36a9909a51d593a3d086c605203859fd5eec0c766d00200afc9a",
    "category_reference.txt": "84f6c3a027cc962a2eee786b9776546d56a9b8edacbaa766f52efd401b7719aa",
    "time1.npy": "c9d9db4d45ce77f7ace9f9a0ee69853c6fd6a17d01e2cf14693e45709b035159",
    "time2.npy": "d27b9430199fa2b481a0d2b3c04a5a25fad5f44e1d48fcc813f26ce91b847469",
}
ROW = re.compile(r"(?P<recording>\d{8}_\d{6})/(?P<frame>\d{10})\.jpg,(?P<label>-?\d+)")


def audit_labels(lines, categories):
    seen = set()
    by_class = Counter()
    class_recordings = defaultdict(set)
    class_dates = defaultdict(set)
    recordings, dates = set(), set()
    for line_number, line in enumerate(lines, 1):
        match = ROW.fullmatch(line)
        if not match:
            raise ValueError(f"invalid annotation identity at line {line_number}")
        recording, frame, label = match["recording"], match["frame"], int(match["label"])
        datetime.strptime(recording, "%Y%m%d_%H%M%S")
        if label not in categories:
            raise ValueError("annotation references an unknown category")
        identity = (recording, int(frame))
        if identity in seen:
            raise ValueError("duplicate frame identity, possibly with conflicting labels")
        seen.add(identity)
        by_class[label] += 1
        class_recordings[label].add(recording)
        class_dates[label].add(recording[:8])
        recordings.add(recording)
        dates.add(recording[:8])
    if not seen:
        raise ValueError("nonempty complete label snapshot required")
    classes = {
        str(label): {
            "name": categories[label],
            "frames": by_class[label],
            "recordings": len(class_recordings[label]),
            "date_prefixes": len(class_dates[label]),
        }
        for label in sorted(categories)
    }
    return {
        "frame_rows": len(seen),
        "recording_count": len(recordings),
        "date_prefix_count": len(dates),
        "inconclusive_frames": by_class[-1],
        "material_labeled_frames": sum(value for label, value in by_class.items() if label != -1),
        "classes": classes,
        "single_recording_material_classes": [
            label
            for label in sorted(categories)
            if label >= 0 and len(class_recordings[label]) == 1
        ],
        "all_material_classes_can_cover_four_recording_partitions": all(
            len(class_recordings[label]) >= 4 for label in categories if label >= 0
        ),
        "independent_sessions_verified": False,
        "recording_names_are_not_subject_or_object_ids": True,
        "recordings": sorted(recordings),
        "date_prefixes": sorted(dates),
    }


def timestamp_summary(first, second):
    streams = []
    for values in (first, second):
        if (
            values.ndim != 1
            or len(values) < 2
            or values.dtype.kind not in "fi"
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError("finite strictly increasing numeric timestamp vectors required")
        streams.append(
            {
                "count": len(values),
                "dtype": str(values.dtype),
                "first_native_units": float(values[0]),
                "last_native_units": float(values[-1]),
                "span_native_units": float(values[-1] - values[0]),
                "median_step_native_units": float(np.median(np.diff(values))),
            }
        )
    return {
        "streams": streams,
        "equal_length": len(first) == len(second),
        "index_offset_abs_max_native_units": float(np.max(np.abs(first - second)))
        if len(first) == len(second)
        else None,
        "physical_synchronization_verified": False,
        "timestamp_units_verified": False,
        "scope": "numeric agreement for one recording only; no media decode or capture-clock proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new metadata audit receipt")
    for name, expected in PINS.items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"metadata snapshot hash mismatch: {name}")
    categories = {}
    for line in (BASE / "category_reference.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(.+):\s*(-?\d+)(?:\s+\(.*\))?", line.strip())
        if not match or int(match[2]) in categories:
            raise ValueError("invalid category reference")
        categories[int(match[2])] = match[1].strip()
    if set(categories) != set(range(-1, 20)):
        raise ValueError("expected the full -1..19 category reference without class filtering")
    result = {
        "status": "NOT_ADMITTED_FOR_FULL_MATERIAL_QUALITY_FIT",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_page": "https://touch-and-go.github.io/",
        "official_folder": "https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B",
        "label_file_id": "1YMK2HfyVzl9ptc5t-A7l2PP-fLhB47kp",
        "input_sha256": PINS,
        "upstream_revision_immutable": False,
        "labels": audit_labels(
            (BASE / "label.txt").read_text(encoding="utf-8").splitlines(), categories
        ),
        "one_recording": {
            "name": "20220319_001819",
            **timestamp_summary(
                np.load(BASE / "time1.npy", allow_pickle=False),
                np.load(BASE / "time2.npy", allow_pickle=False),
            ),
        },
        "dataset_listing": {"returned_folders": 100, "complete": False, "provider_page_cap": 100},
        "media_downloaded": False,
        "model_loaded": False,
        "training": False,
        "quality_evaluated": False,
        "sealed_audit_opened": False,
        "limitations": [
            "labels inspected for feasibility; future use is observed development data",
            "recordings and date prefixes are not verified independent people/objects/sessions",
            "one material has only one recording and cannot span independent train and evaluation",
            "do not split neighboring frames or remove the rare class to raise metrics",
            "both-input necessity and tactile-model suitability remain unmeasured",
            "official CC BY 4.0 project declaration is not a per-asset ownership verification",
            "no upstream code license was established and no upstream code was executed",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        result["status"],
        result["labels"]["frame_rows"],
        "frames",
        result["labels"]["recording_count"],
        "recordings",
    )


if __name__ == "__main__":
    main()
