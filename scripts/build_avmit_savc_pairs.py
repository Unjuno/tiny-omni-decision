"""Build balanced AVMIT SAVC pairs from one non-sealed split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tiny_omni_decision.avmit_metadata import build_savc_pair_rows, canonical_jsonl_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "calibration", "validation"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs, leftovers = build_savc_pair_rows(rows, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_jsonl_bytes(pairs)
    args.output.write_bytes(payload)
    report = {
        "split": args.split,
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "pair_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "source_video_count": len(rows),
        "pair_groups": len(pairs) // 8,
        "decision_rows": len(pairs),
        "positive_rows": sum(row["target"] == "same_action" for row in pairs),
        "negative_rows": sum(row["target"] == "different_action" for row in pairs),
        "unpaired_source_video_hashes": leftovers,
        "unpaired_count": len(leftovers),
        "grouping": (
            "four distinct source videos per pair group: two clips from each of two labels; "
            "same-label positives are cross-clip and no audio/video row uses the same clip"
        ),
        "unimodal_balance": (
            "each source audio and source video appears once under each target within "
            "its pair group"
        ),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
