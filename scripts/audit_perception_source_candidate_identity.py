"""Audit train-corpus collisions after removing event-specific option IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def audit_records(records):
    groups = defaultdict(list)
    candidate_count = held_count = duplicate_count = 0
    partial_positive_crop_records = 0
    for record in records:
        if record.get("status") != "candidate":
            held_count += record.get("status") == "held"
            duplicate_count += record.get("status") == "duplicate"
            continue
        candidate_count += 1
        request = record["request"]
        candidate_fingerprints = {}
        for candidate in request["candidates"]:
            candidate_id = candidate["candidate_id"]
            fingerprint = _sha256(candidate["frame_boxes"])
            if candidate_id in candidate_fingerprints:
                raise ValueError(f"duplicate candidate ID in {record['record_id']}")
            candidate_fingerprints[candidate_id] = fingerprint
        candidate_counts = defaultdict(int)
        target_counts = defaultdict(int)
        for fingerprint in candidate_fingerprints.values():
            candidate_counts[fingerprint] += 1

        supervision = record["supervision"]
        if supervision["none_visible"]:
            if supervision["target_candidate_ids"]:
                raise ValueError(f"none-visible row has object targets: {record['record_id']}")
            targets = ["none_visible"]
        else:
            targets = sorted(
                candidate_fingerprints[candidate_id]
                for candidate_id in supervision["target_candidate_ids"]
            )
            for fingerprint in targets:
                target_counts[fingerprint] += 1
            if not targets:
                raise ValueError(f"visible row has no target candidates: {record['record_id']}")
        partial_positive_crop_records += any(
            0 < target_counts[fingerprint] < count
            for fingerprint, count in candidate_counts.items()
        )

        signature = {
            "task": request["task"],
            "instruction": request["instruction"],
            "video_id": record["media_reference"]["video_id"],
            "audio_interval_us": request["audio_interval_us"],
            "candidate_frame_box_fingerprints": sorted(candidate_fingerprints.values()),
        }
        input_hash = _sha256(signature)
        groups[input_hash].append(
            {
                "record_id": record["record_id"],
                "targets": targets,
                "candidate_count": len(candidate_fingerprints),
            }
        )

    repeated = [rows for rows in groups.values() if len(rows) > 1]
    contradictory = [
        rows for rows in repeated if len({_canonical(row["targets"]) for row in rows}) > 1
    ]
    return {
        "protocol": "perception_test_source_candidate_input_identity_v1",
        "candidate_records": candidate_count,
        "held_records": held_count,
        "duplicate_records": duplicate_count,
        "partial_positive_crop_records": partial_positive_crop_records,
        "unique_model_inputs": len(groups),
        "repeated_input_groups": len(repeated),
        "repeated_input_records": sum(map(len, repeated)),
        "contradictory_input_groups": len(contradictory),
        "contradictory_records": sum(map(len, contradictory)),
        "contradictions": [
            {
                "record_ids": sorted(row["record_id"] for row in rows),
                "candidate_count": rows[0]["candidate_count"],
                "target_fingerprints": sorted(
                    {_canonical(row["targets"]) for row in rows}
                ),
            }
            for rows in contradictory
        ],
        "identity_scope": (
            "within-video exact interval and normalized candidate frame boxes; "
            "candidate IDs and order are ignored"
        ),
        "limitations": [
            "uses train annotations only",
            "does not compare pixel-level crop hashes across different video IDs",
            "does not establish source-label semantics",
        ],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_file(source_path):
    source_path = Path(source_path)
    with source_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = audit_records(rows)
    report["source_split"] = "official train only"
    report["source_corpus_sha256"] = sha256_file(source_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior identity report: {output}")
    report = audit_file(args.corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
