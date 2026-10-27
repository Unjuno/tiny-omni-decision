"""Stream a pinned train label object and count in-scope decision support."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import requests

from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

EXPECTED_SHA = "a97b079ccccf635a980bfe4b8180a0c844b4d9c887b2b7bc56535f539af044ea"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_FULL_TRAIN_SUPPORT_PROTOCOL_20261027.md")


def count_support(groups, scope):
    counts = {label: Counter() for label in scope}
    pairs = Counter()
    images = eligible_images = eligible_pairs = 0
    for group in groups:
        images += 1
        positives = group["positive_labels"] & scope
        negatives = group["negative_labels"] & scope
        if positives & negatives:
            raise ValueError("contradictory verification labels")
        for label in positives:
            counts[label]["positive_images"] += 1
        for label in negatives:
            counts[label]["negative_images"] += 1
        if len(negatives) >= 3 and positives:
            eligible_images += 1
            eligible_pairs += len(positives)
            for label in positives:
                counts[label]["eligible_target_images"] += 1
                for negative in negatives:
                    pairs[(label, negative)] += 1
    return images, eligible_images, eligible_pairs, counts, pairs


def stream_lines(response, state):
    pending = b""
    for block in response.iter_content(chunk_size=1024 * 1024):
        state["sha"].update(block)
        state["bytes"] += len(block)
        if state["bytes"] > TOTAL:
            raise ValueError("source exceeds expected length")
        data = pending + block
        lines = data.split(b"\n")
        pending = lines.pop()
        for line in lines:
            yield line.decode("utf-8-sig")
        if state["bytes"] // (100 * 1024 * 1024) > state["logged"]:
            state["logged"] = state["bytes"] // (100 * 1024 * 1024)
            print(f"read {state['bytes']} bytes", file=sys.stderr, flush=True)
    if pending:
        yield pending.decode("utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(TARGETS) != TARGET_SHA:
        raise ValueError("output exists or frozen scope mismatch")
    source = json.loads(TARGETS.read_bytes())
    names = {item["label_id"]: item["display_name"] for item in source["selected_target_classes"]}
    if len(names) != 11:
        raise ValueError("unexpected class count")
    state = {"sha": hashlib.sha256(), "bytes": 0, "logged": 0}
    headers = {"If-Match": f'"{ETAG}"', "Accept-Encoding": "identity"}
    with requests.get(URL, headers=headers, stream=True, timeout=(15, 90)) as response:
        if (response.status_code != 200 or response.headers.get("ETag", "").strip('"') != ETAG
                or response.headers.get("Content-Length") != str(TOTAL)
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("source response identity or length mismatch")
        groups = group_verification_labels(csv.DictReader(stream_lines(response, state)))
        images, eligible_images, eligible_pairs, counts, pairs = count_support(groups, set(names))
    source_sha = state["sha"].hexdigest()
    if state["bytes"] != TOTAL or source_sha != EXPECTED_SHA:
        raise ValueError(f"source mismatch: bytes={state['bytes']} sha256={source_sha}")
    result = {
        "status": "PASS_FULL_TRAIN_LABEL_AUDIT", "source_bytes": state["bytes"],
        "source_sha256": source_sha, "etag": ETAG, "targets_sha256": TARGET_SHA,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "image_groups": images, "eligible_unique_images": eligible_images,
        "eligible_target_image_pairs": eligible_pairs,
        "class_counts": {names[k]: dict(counts[k]) for k in sorted(names)},
        "available_negative_pairs": {
            names[k]: {names[n]: pairs[(k, n)] for n in sorted(names) if n != k}
            for k in sorted(names)
        },
        "limitations": (
            "train labels only; availability is not sampled role balance; "
            "media rights and splits unverified"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
