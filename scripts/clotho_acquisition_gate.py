"""Fail closed before consuming the full frozen AQA development acquisition."""

import hashlib
import json

from scripts.clotho_media_identity import ALIAS_SHA256, media_basename
from scripts.fetch_clotho_development_media import HASHES


def validate_records(report, expected):
    """Validate exact source coverage and provenance, independent of report order."""
    if report.get("status") != "PASS":
        raise ValueError("acquisition is not terminal PASS")
    records = report["files"]
    if len(records) != len(expected):
        raise ValueError("incomplete or extra acquisition records")
    seen, digest_splits = set(), {}
    for record in records:
        identity = record["source_id"]
        if identity in seen or identity not in expected:
            raise ValueError("duplicate or unexpected source")
        seen.add(identity)
        wanted = expected[identity]
        for key in ("split", "license", "uploader"):
            if record[key] != wanted[key]:
                raise ValueError("acquisition provenance mismatch")
        if record["member"] != "audio_files/" + media_basename(wanted):
            raise ValueError("archive member mismatch")
        digest = record["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid SHA256")
        if identity == "16854" and digest != ALIAS_SHA256:
            raise ValueError("alias content mismatch")
        if digest in digest_splits and digest_splits[digest] != record["split"]:
            raise ValueError("identical media bytes cross splits")
        digest_splits[digest] = record["split"]
    return sorted(records, key=lambda r: r["source_id"])


def complete_acquisition(media, manifests):
    expected = {}
    for split, digest in HASHES.items():
        data = (manifests / f"{split}.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("frozen manifest hash mismatch")
        for row in json.loads(data):
            identity = row["source_id"]
            if identity in expected and expected[identity]["split"] != split:
                raise ValueError("source overlaps frozen splits")
            expected[identity] = dict(row, split=split)
    report = json.loads((media / "result.json").read_text())
    if report.get("split_hashes") != HASHES:
        raise ValueError("acquisition refers to different splits")
    records = validate_records(report, expected)
    for row in records:
        content = (media / f"{row['source_id']}.wav").read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("saved media bytes differ")
    return records
