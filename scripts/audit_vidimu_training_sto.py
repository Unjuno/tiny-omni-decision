"""Validate all frozen training quaternion records; never read held-out values."""

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.quaternion_sto import parse_quaternion_sto


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = root / "vidimu-frozen-split-01/split.json"
    archive_path = root / "vidimu-source-audit-02/dataset.zip"
    if sha(split_path) != "230e4f9f9fb6218b90fea0894396bc96bf67a0951864301d0fe289f4a0457b2e":
        raise ValueError("split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha(archive_path) != split["archive_sha256"]:
        raise ValueError("archive changed")
    output = root / "vidimu-training-sto-audit-01"
    output.mkdir(exist_ok=False)
    results = []
    layouts = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for record in split["records"]["train"]:
            member = record["member"].removesuffix(".raw") + ".sto"
            item = {**record, "sto_member": member}
            try:
                content = archive.read(member)
                item["sha256"] = hashlib.sha256(content).hexdigest()
                parsed = parse_quaternion_sto(content.decode())
                times = parsed["timestamps_ms"]
                deltas = np.diff(times)
                item.update({"status": "pass", "shape": list(parsed["values"].shape),
                             "sensors": parsed["sensors"], "start_ms": float(times[0]),
                             "end_ms": float(times[-1]), "min_delta_ms": float(deltas.min()),
                             "max_delta_ms": float(deltas.max())})
                layouts["|".join(parsed["sensors"])] += 1
            except (KeyError, ValueError, UnicodeError) as error:
                item.update({"status": "fail", "error": str(error)})
            results.append(item)
    summary = {"records": len(results), "status_counts": dict(Counter(
        r["status"] for r in results)), "layouts": dict(layouts),
        "validation_or_audit_payload_read": False, "records_detail": results}
    write(output / "result.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records_detail"}))


if __name__ == "__main__":
    main()
