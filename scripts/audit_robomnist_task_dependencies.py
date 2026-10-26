"""Audit shared-asset dependencies in fixed train/validation manifests only."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.extract_perception_temporal_match_efficientat import sha256_file

BASE = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-correspondence-seed17-attempt-05")
HASHES = {
    "train": "0aa00362dc933c85926cd11bde8774ce96d1d350468e5927f226f04628b96049",
    "validation": "02175d41fb7434fe453510bacfc55ea25f0d8d5af34e84cb0c9d01d2482f1e21",
}


def dependency_accounting(rows):
    parents = {}

    def root(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = root(parents[value])
        return parents[value]

    for row in rows:
        refs = row["source_refs"]
        groups = [refs["video"]["group_id"]] + [
            item["group_id"] for item in refs["audio_candidates"]
        ]
        for group in groups[1:]:
            parents[root(group)] = root(groups[0])
    counts = Counter(root(group) for group in parents)
    return {
        "rows": len(rows),
        "unique_repetitions": len(parents),
        "asset_sharing_connected_components": len(counts),
        "component_sizes": dict(sorted(Counter(counts.values()).items())),
        "independent_sessions_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new dependency receipt")
    result = {"sealed_audit_opened": False, "training": False, "quality_evaluated": False}
    for split, expected in HASHES.items():
        path = BASE / f"{split}.json"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("fixed development manifest hash mismatch")
        result[split] = {
            **dependency_accounting(json.loads(path.read_bytes())),
            "manifest_sha256": actual,
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
