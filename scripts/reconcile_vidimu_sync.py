"""Reconcile synchronization metadata only, without opening held-out payloads."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = root / "vidimu-frozen-split-01/split.json"
    sync_path = root / "vidimu-analysis-audit-02/infoToSync.csv"
    pins = {split_path: "230e4f9f9fb6218b90fea0894396bc96bf67a0951864301d0fe289f4a0457b2e",
            sync_path: "4596803e90e2908717ab227846f7a1c10a5c6b29c9bcbddb90e940a7259c519f"}
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("input pin mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    with sync_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["Subject"], row["Activity"], row["Trial"])
            kind = row["Type"]
            if kind in grouped[key] or kind not in {"raw", "mot", "mp4", "csv"}:
                raise ValueError("duplicate or unknown offset type")
            offset = int(row["CutFrames"])
            if not 0 < offset < 15:
                raise ValueError("unexpected offset")
            grouped[key][kind] = offset
    known = {(r["subject"], r["activity"], r["trial"])
             for records in split["records"].values() for r in records}
    if grouped.keys() - known:
        raise ValueError("offset refers to an unknown trial")
    records, counts = [], {}
    for group, items in split["records"].items():
        count = Counter()
        for record in items:
            key = (record["subject"], record["activity"], record["trial"])
            offsets = grouped.get(key, {})
            if offsets.get("raw") != offsets.get("mot"):
                raise ValueError("sensor offsets disagree")
            if offsets.get("mp4") != offsets.get("csv"):
                raise ValueError("video offsets disagree")
            status = ("missing_not_assumed_zero" if not offsets else
                      "ambiguous_both_positive" if "raw" in offsets and "mp4" in offsets
                      else "single_direction_recorded")
            records.append({**record, "split": group, "offsets": offsets, "status": status})
            count[status] += 1
        counts[group] = dict(count)
    output = root / "vidimu-sync-reconciliation-01"
    output.mkdir(exist_ok=False)
    write(output / "result.json", {"counts": counts, "records": records,
                                   "payloads_read": False, "physical_sync_verified": False})
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
