"""Inventory pinned verification labels for already page-matched train images."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import requests

from scripts.audit_openimages_full_train_support import EXPECTED_SHA, stream_lines
from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

TRAIN = Path("C:/CodexArtifacts/pretrained-reuse-r2/openimages-v6-support-audit-20261026") / (
    "page-matched-train-tasks.json"
)
FEATURES = Path("C:/CodexArtifacts/pretrained-reuse-r2") / (
    "openimages-videoprism-features-20261026-attempt-01/train-rows.jsonl"
)
TRAIN_SHA = "71e3c95f26fdce7a5c6128e0dcbe81b47ab0f4139bf0dcc05baf341a9f374515"
FEATURES_SHA = "2cc95cfa92369938611a4519801b332b6d3c623bdb1414cb9f16df0e7e003309"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_EXISTING_TRAIN_RELABEL_PROTOCOL_20261027.md")


def inventory(groups, examples, feature_ids, scope):
    wanted = {row["image_id"]: row for row in examples}
    if len(wanted) != len(examples) or any(row["split"] != "train" for row in examples):
        raise ValueError("train images must be unique and train-only")
    found = {}
    for group in groups:
        image_id = group["image_id"]
        if image_id not in wanted:
            continue
        if image_id in found:
            raise ValueError("duplicate group")
        positive = group["positive_labels"] & scope
        negative = group["negative_labels"] & scope
        row = wanted[image_id]
        if positive & negative or row["target_label"] not in positive:
            raise ValueError("target is not an independent verified positive")
        found[image_id] = {
            "image_id": image_id,
            "target_label": row["target_label"],
            "positive_labels": sorted(positive),
            "negative_labels": sorted(negative),
            "eligible": len(negative) >= 3,
            "feature_available": image_id in feature_ids,
        }
    if set(found) != set(wanted):
        raise ValueError("not all fixed train images found")
    return [found[image_id] for image_id in sorted(found)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(TRAIN) != TRAIN_SHA or digest(FEATURES) != FEATURES_SHA:
        raise ValueError("output exists or pinned input mismatch")
    if digest(TARGETS) != TARGET_SHA:
        raise ValueError("scope mismatch")
    examples = json.loads(TRAIN.read_bytes())["examples"]
    scope_data = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    scope = {row["label_id"] for row in scope_data}
    names = {row["label_id"]: row["display_name"] for row in scope_data}
    feature_ids = {json.loads(line)["image_id"] for line in FEATURES.read_text().splitlines()}
    state = {"sha": hashlib.sha256(), "bytes": 0, "logged": 0}
    with requests.get(URL, headers={"If-Match": f'"{ETAG}"', "Accept-Encoding": "identity"},
                      stream=True, timeout=(15, 90)) as response:
        if (response.status_code != 200 or response.headers.get("ETag", "").strip('"') != ETAG
                or response.headers.get("Content-Length") != str(TOTAL)
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("source identity mismatch")
        groups = group_verification_labels(csv.DictReader(stream_lines(response, state)))
        rows = inventory(groups, examples, feature_ids, scope)
    if state["bytes"] != TOTAL or state["sha"].hexdigest() != EXPECTED_SHA:
        raise ValueError("source hash mismatch")
    counts = Counter((names[row["target_label"]], row["feature_available"])
                     for row in rows if row["eligible"])
    result = {
        "source_sha256": EXPECTED_SHA, "train_sha256": TRAIN_SHA,
        "features_sha256": FEATURES_SHA, "scope_sha256": TARGET_SHA,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "total": len(rows), "eligible": sum(row["eligible"] for row in rows),
        "eligible_with_features": sum(row["eligible"] and row["feature_available"] for row in rows),
        "eligible_by_target": {name: {"with_features": counts[(name, True)],
                                       "without_features": counts[(name, False)]}
                               for name in names.values()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
