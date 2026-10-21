"""Inspect one preselected training record, without claiming synchronized data."""

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = root / "vidimu-frozen-split-01/split.json"
    if sha(split_path) != "230e4f9f9fb6218b90fea0894396bc96bf67a0951864301d0fe289f4a0457b2e":
        raise ValueError("split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    record = split["records"]["train"][0]
    archive_path = root / "vidimu-source-audit-02/dataset.zip"
    if sha(archive_path) != split["archive_sha256"]:
        raise ValueError("archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        content = archive.read(record["member"])
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline=None)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["QUAT"]].append([float(row[k]) for k in ("w", "x", "y", "z", "timestamp")])
    summary = {}
    for name, values in grouped.items():
        array = np.asarray(values)
        times = array[:, 4]
        positive = times[times > 0]
        delta = np.diff(positive)
        summary[name] = {
            "rows": len(array), "finite": bool(np.isfinite(array).all()),
            "zero_timestamp_rows": int((times == 0).sum()),
            "positive_duration_seconds": float(positive[-1] - positive[0]),
            "nonincreasing_positive_deltas": int((delta <= 0).sum()),
            "delta_seconds_quantiles": np.quantile(delta, [0, .5, .95, 1]).tolist(),
            "quaternion_norm_quantiles": np.quantile(
                np.linalg.norm(array[:, :4], axis=1), [0, .5, 1]
            ).tolist(),
        }
    out = root / "vidimu-training-format-audit-01"
    out.mkdir(exist_ok=False)
    import hashlib

    result = {"record": record, "member_sha256": hashlib.sha256(content).hexdigest(),
              "split_sha256": sha(split_path), "fields": list(rows[0]),
              "sensor_rows": dict(Counter(row["QUAT"] for row in rows)),
              "summary": summary, "alignment_verified": False,
              "validation_or_audit_payload_read": False}
    write(out / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
