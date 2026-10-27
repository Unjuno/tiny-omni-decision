"""Measure same-option quartet availability on pinned train metadata IDs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_openimages_balanced_quartets import build_quartets
from scripts.audit_openimages_full_train_support import EXPECTED_SHA
from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.audit_openimages_train_metadata_join import (
    META_ETAG,
    META_SHA,
    META_SIZE,
    META_URL,
    metadata_ids,
    open_rows,
)
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_FULL_TRAIN_QUARTET_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def filter_candidates(groups, train_ids, scope):
    candidates = []
    for group in groups:
        if group["image_id"] not in train_ids:
            continue
        positive = group["positive_labels"] & scope
        negative = group["negative_labels"] & scope
        if positive & negative:
            raise ValueError("conflicting verified labels")
        if positive and len(negative) >= 3:
            candidates.append({"image_id": group["image_id"],
                               "positive_labels": sorted(positive),
                               "negative_labels": sorted(negative)})
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(REPO) or digest(TARGETS) != TARGET_SHA:
        raise ValueError("new external output and pinned scope required")
    scope_data = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    names = {row["label_id"]: row["display_name"] for row in scope_data}
    if len(names) != 11:
        raise ValueError("unexpected scope")
    ids = open_rows(META_URL, META_ETAG, META_SIZE, META_SHA, metadata_ids)
    candidates = open_rows(URL, ETAG, TOTAL, EXPECTED_SHA,
                           lambda rows: filter_candidates(group_verification_labels(rows),
                                                          ids, set(names)))
    if len(candidates) != 3377:
        raise ValueError("candidate total differs from pinned joined source census")
    saved = {row["image_id"] for row in candidates}
    quartets, option_sets, complete, unused = build_quartets(candidates, saved, set(names))
    targets = Counter(row["target"] for group in quartets for row in group["rows"])
    result = {
        "status": "TRAIN_METADATA_FEASIBILITY_ONLY", "metadata_ids": len(ids),
        "eligible_images": len(candidates), "possible_option_sets": option_sets,
        "four_target_option_sets": complete, "disjoint_quartets": len(quartets),
        "selected_images": len(quartets) * 4, "unused_images": unused,
        "target_counts": {names[k]: targets[k] for k in names},
        "metadata_sha256": META_SHA, "labels_sha256": EXPECTED_SHA,
        "scope_sha256": TARGET_SHA, "protocol_sha256": digest(PROTOCOL),
        "script_sha256": digest(Path(__file__)),
        "limitations": "greedy train-only aggregate; no individual rights or split proof",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
