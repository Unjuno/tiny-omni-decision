"""Read only frozen training annotation members; no media or held-out payloads."""

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = root / "starss23-room-splits-v1/train.jsonl"
    metadata = root / "starss23-source-audit-01/metadata_dev.zip"
    pins = {split_path: "571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978",
            metadata: "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6"}
    for path, digest in pins.items():
        if sha256_file(path) != digest:
            raise ValueError("pinned input changed")
    records = [json.loads(line) for line in split_path.read_text().splitlines()]
    if len(records) != 49 or any(r["split"] != "train" for r in records):
        raise ValueError("training-only scope changed")
    out = root / "starss23-spatial-training-audit-01"
    out.mkdir(exist_ok=False)
    labels, angles, elevations, distances, details = Counter(), set(), set(), [], []
    with zipfile.ZipFile(metadata) as archive:
        for record in records:
            frames = Counter()
            rows = list(csv.reader(io.StringIO(archive.read(record["member"]).decode())))
            for row in rows:
                if len(row) != 6:
                    raise ValueError("expected six fields")
                tick, label, source, azimuth, elevation, distance = map(int, row)
                if (tick < 0 or not 0 <= label <= 12 or source < 0
                        or not -180 <= azimuth <= 180 or not -90 <= elevation <= 90
                        or distance < 0):
                    raise ValueError("invalid spatial annotation")
                frames[tick] += 1
                labels[label] += 1
                angles.add(azimuth)
                elevations.add(elevation)
                distances.append(distance)
            details.append(dict(recording=record["clip"], rows=len(rows), active_ticks=len(frames),
                                overlapping_ticks=sum(n > 1 for n in frames.values())))
    report = dict(scope="49 frozen training recordings only", input_hashes={str(k): v
                  for k, v in pins.items()}, script_sha256=sha256_file(Path(__file__)),
                  raw_class_annotation_rows=dict(labels), annotation_rows=sum(labels.values()),
                  distinct_azimuth_degrees=len(angles), distinct_elevation_degrees=len(elevations),
                  distance_cm_range=[min(distances), max(distances)],
                  zero_distance_rows=sum(d == 0 for d in distances), recordings=details,
                  fields=["tick_100ms", "raw_class", "source_index", "azimuth_deg",
                          "elevation_deg", "distance_cm"],
                  visibility_annotation_present=False, bounding_box_annotation_present=False,
                  note="six-column schema has no visibility or visual-object box fields",
                  media_read=False, calibration_read=False, validation_read=False,
                  sealed_audit_read=False)
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "recordings"}))


if __name__ == "__main__":
    main()
