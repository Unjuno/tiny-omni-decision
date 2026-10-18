"""Fail-closed verification before using acquired STARSS23 media for features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.fetch_starss23_frozen_media import load_splits


def validate_acquisition(report, expected_rows, verify_file=sha256_file):
    if report.get("status") != "PASS":
        raise ValueError("acquisition is incomplete")
    expected = {r["clip"]: r for r in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("duplicate expected recording")
    pairs = report.get("pairs", [])
    if len(pairs) != len(expected) or {r["clip"] for r in pairs} != set(expected):
        raise ValueError("recording set differs from frozen splits")
    identities = {}
    for pair in pairs:
        source = expected[pair["clip"]]
        for key in ("split", "room", "upstream_split", "source_revision"):
            if pair.get(key) != source[key]:
                raise ValueError(f"identity mismatch: {key}")
        if set(pair["files"]) != {"audio", "video"}:
            raise ValueError("paired audio/video required")
        for kind, media in pair["files"].items():
            path = Path(media["path"])
            if not path.is_file() or path.stat().st_size != media["bytes"]:
                raise ValueError("missing media or byte-size mismatch")
            checksum = verify_file(path)
            if checksum != media["sha256"]:
                raise ValueError("media hash mismatch")
            identity = (kind, checksum)
            if identity in identities:
                raise ValueError("duplicate underlying media across recordings")
            identities[identity] = pair["clip"]
    return {
        "status": "PASS",
        "recordings": len(pairs),
        "unique_media_assets": len(identities),
        "rooms": len({r["room"] for r in pairs}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.progress.read_text())
    if report.get("summary_sha256") != sha256_file(args.summary):
        raise ValueError("acquisition used a different split summary")
    summary = json.loads(args.summary.read_text())
    rows = load_splits(args.splits, summary)
    result = validate_acquisition(report, rows)
    result["acquisition_sha256"] = sha256_file(args.progress)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
