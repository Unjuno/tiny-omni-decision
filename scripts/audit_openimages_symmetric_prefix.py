"""Bounded train-only feasibility screen for symmetric Open Images options."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import requests

import tiny_omni_decision.openimages_options as options_module
from scripts.audit_openimages_target_scope_shortcut import TARGET_SHA, TARGETS
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv?generation=1580742923196487"
ETAG = "a0c8491396bfa736bec85cb172180bd3"
SIZE = 67_108_864
TOTAL = 2_474_671_021
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_SYMMETRIC_PREFIX_PROTOCOL_20261027.md")


def count_groups(groups, scope):
    """Count opportunities per target; ignore incomplete last image group."""
    if len(scope) != 11:
        raise ValueError("unexpected frozen class count")
    counts = {label: Counter() for label in scope}
    groups_seen, prior = 0, None
    for group in groups:
        if prior is not None:
            groups_seen += 1
            positive = prior["positive_labels"] & scope
            negative = prior["negative_labels"] & scope
            for label in positive:
                n = len(negative - {label})
                counts[label]["positive_records"] += 1
                counts[label][f"negative_count_{min(n, 3)}"] += 1
                if n >= 3:
                    counts[label]["four_choice_eligible"] += 1
        prior = group
    if prior is None:
        raise ValueError("empty label prefix")
    return groups_seen, counts


def fetch_prefix(session):
    headers = {
        "Range": f"bytes=0-{SIZE - 1}",
        "If-Match": f'"{ETAG}"',
        "Accept-Encoding": "identity",
    }
    with session.get(URL, headers=headers, stream=True, timeout=(15, 60)) as response:
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != f"bytes 0-{SIZE - 1}/{TOTAL}"
            or response.headers.get("ETag", "").strip('"') != ETAG
        ):
            raise ValueError("server did not honor pinned range and identity")
        data = bytearray()
        for block in response.iter_content(chunk_size=1024 * 1024):
            data.extend(block)
            if len(data) > SIZE:
                raise ValueError("range response exceeded requested bytes")
        if len(data) != SIZE:
            raise ValueError("range response incomplete")
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(TARGETS) != TARGET_SHA:
        raise ValueError("output exists or frozen scope mismatch")
    if (
        Path(options_module.__file__).resolve()
        != Path("src/tiny_omni_decision/openimages_options.py").resolve()
    ):
        raise ValueError("wrong options module; set PYTHONPATH=src;.")
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_hash != "eae4d247241d716ccd3f8f4ebb24249e4b0901f54536aededfaa5a4613749d78":
        raise ValueError("protocol hash mismatch")
    source = json.loads(TARGETS.read_bytes())
    names = {item["label_id"]: item["display_name"] for item in source["selected_target_classes"]}
    data = fetch_prefix(requests.Session())
    prefix_sha = hashlib.sha256(data).hexdigest()
    if b"\n" not in data:
        raise ValueError("no complete CSV row")
    complete = data[: data.rfind(b"\n") + 1]
    reader = csv.DictReader(io.StringIO(complete.decode("utf-8-sig")))
    groups_seen, counts = count_groups(group_verification_labels(reader), set(names))
    result = {
        "status": "PASS_BOUNDED_TRAIN_ONLY_PREFIX",
        "generation": "1580742923196487",
        "etag": ETAG,
        "range_bytes": SIZE,
        "prefix_sha256": prefix_sha,
        "source_total_bytes": TOTAL,
        "complete_image_groups": groups_seen,
        "targets_sha256": TARGET_SHA,
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": protocol_hash,
        "counts": {names[label]: dict(counts[label]) for label in sorted(names)},
        "scope": "nonrandom train prefix only; full-source hash, rights and media unverified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
