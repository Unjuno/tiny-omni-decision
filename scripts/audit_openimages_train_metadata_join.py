"""Join pinned train image metadata to the full verified-label census."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import requests

from scripts.audit_openimages_full_train_support import EXPECTED_SHA, count_support
from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

META_URL = ("https://storage.googleapis.com/openimages/2018_04/train/"
            "train-images-boxable-with-rotation.csv?generation=1526567626302755")
META_ETAG = "a4ac0bcedb5c2df4d1b1230dc6b41b8f"
META_SIZE = 638407721
META_SHA = "05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_TRAIN_METADATA_JOIN_PROTOCOL_20261027.md")


def metadata_ids(rows):
    ids = set()
    for row in rows:
        image_id = row.get("ImageID")
        if not image_id or image_id in ids:
            raise ValueError("missing or duplicate metadata image ID")
        if row.get("Subset") != "train" or row.get("License", "").rstrip("/") != (
            "https://creativecommons.org/licenses/by/2.0"
        ):
            raise ValueError("unexpected train subset or declared license")
        ids.add(image_id)
    return ids


def stream_lines(response, state, size):
    pending = b""
    for block in response.iter_content(chunk_size=1024 * 1024):
        state["sha"].update(block)
        state["bytes"] += len(block)
        if state["bytes"] > size:
            raise ValueError("source length exceeded")
        lines = (pending + block).split(b"\n")
        pending = lines.pop()
        for line in lines:
            yield line.decode("utf-8-sig")
        logged = state["bytes"] // (100 * 1024 * 1024)
        if logged > state["logged"]:
            state["logged"] = logged
            print(f"read {state['bytes']} / {size}", file=sys.stderr, flush=True)
    if pending:
        yield pending.decode("utf-8-sig")


def open_rows(url, etag, size, expected_sha, consumer):
    state = {"sha": hashlib.sha256(), "bytes": 0, "logged": 0}
    with requests.get(url, headers={"If-Match": f'"{etag}"', "Accept-Encoding": "identity"},
                      stream=True, timeout=(15, 90)) as response:
        if (response.status_code != 200 or response.headers.get("ETag", "").strip('"') != etag
                or response.headers.get("Content-Length") != str(size)
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("source identity mismatch")
        value = consumer(csv.DictReader(stream_lines(response, state, size)))
    if state["bytes"] != size or state["sha"].hexdigest() != expected_sha:
        raise ValueError("source hash mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(TARGETS) != TARGET_SHA:
        raise ValueError("output exists or frozen scope mismatch")
    scope_data = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    names = {item["label_id"]: item["display_name"] for item in scope_data}
    if len(names) != 11:
        raise ValueError("unexpected class count")
    ids = open_rows(META_URL, META_ETAG, META_SIZE, META_SHA, metadata_ids)

    def count(rows):
        groups = group_verification_labels(rows)
        return count_support((group for group in groups if group["image_id"] in ids), set(names))

    images, eligible, pairs_total, counts, pairs = open_rows(URL, ETAG, TOTAL, EXPECTED_SHA, count)
    result = {
        "status": "PASS_TRAIN_METADATA_JOIN", "metadata_sha256": META_SHA,
        "label_sha256": EXPECTED_SHA, "scope_sha256": TARGET_SHA,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "metadata_image_ids": len(ids), "annotated_train_image_groups": images,
        "eligible_unique_images": eligible, "eligible_target_image_pairs": pairs_total,
        "class_counts": {names[k]: dict(counts[k]) for k in sorted(names)},
        "available_negative_pairs": {
            names[k]: {names[n]: pairs[(k, n)] for n in sorted(names) if n != k}
            for k in sorted(names)
        },
        "limitations": "declarations only; individual rights, source splits, and role balance open",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
