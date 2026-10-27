"""Freeze a stratified train-source sample for sound annotation review."""

import argparse
import hashlib
import importlib.util
import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.audit_source_annotation_semantics import BUILDER, BUILDER_SHA, audit_record
from scripts.intervene_source_same_label_audio import ARCHIVE, ARCHIVE_SHA
from scripts.train_source_unimodal_cv import LABELS, PINNED, ROOT, digest

MEDIA = ROOT / "perception-av-event-train-media-01"
MANIFEST = MEDIA / "manifest.json"
MANIFEST_SHA = "5ac15ab7ef7efda29a803f783b4a3a13fd3895ebaecbdf9c9c7e0282b9195918"
PROTOCOL = Path("docs/pretrained_reuse/R2_SOURCE_SEMANTICS_REVIEW_PROTOCOL_20261027.md")
STRATA = ("none_related_candidate", "none_other", "visible_one", "visible_multi")


def choose(rows, count=4):
    rows = list(rows)
    selected, used = [], set()
    for category in STRATA:
        candidates = sorted(
            (r for r in rows if r["category"] == category),
            key=lambda r: (
                hashlib.sha256(f"source-label-review-v1:17:{r['record_id']}".encode()).digest(),
                r["record_id"],
            ),
        )
        chosen = []
        for row in candidates:
            if row["video_id"] not in used:
                used.add(row["video_id"])
                chosen.append(row)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            raise ValueError("not enough distinct videos")
        selected.extend(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_sha != "5f0457f8c97904500bdcf17e4b5ffeadb6838fef6cb970f87a3d9ff10c5d8424":
        raise ValueError("protocol hash mismatch")
    for path, expected in (
        (ARCHIVE, ARCHIVE_SHA),
        (LABELS, PINNED[LABELS]),
        (BUILDER, BUILDER_SHA),
        (MANIFEST, MANIFEST_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"pinned hash mismatch: {path}")
    manifest = json.loads(MANIFEST.read_bytes())
    clips = {c["metadata"]["video_id"]: c for c in manifest["clips"]}
    if len(clips) != len(manifest["clips"]):
        raise ValueError("duplicate media video identity")
    spec = importlib.util.spec_from_file_location("source_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    records = list(map(json.loads, LABELS.read_bytes().splitlines()))
    with zipfile.ZipFile(ARCHIVE) as archive:
        source = json.loads(archive.read("all_train.json"))
    entries = []
    for record in records:
        video = record["group_id"]
        audit = audit_record(record, source[video], builder)
        if video not in clips or clips[video]["metadata"]["split"] != "train":
            raise ValueError("missing or non-train media")
        category = (
            "none_related_candidate"
            if audit["none"] and audit["known_parent_candidates"]
            else "none_other"
            if audit["none"]
            else "visible_multi"
            if audit["known_parent_candidates"] > 1
            else "visible_one"
        )
        entries.append(
            {
                "record_id": record["record_id"],
                "video_id": video,
                "category": category,
                "audio_interval_us": record["provenance"]["audio_interval_us"],
                "event_id": record["provenance"]["sound_annotation_id"],
            }
        )
    if len(entries) != 2593 or len({r["record_id"] for r in entries}) != 2593:
        raise ValueError("record coverage mismatch")
    selected = choose(entries)
    for row in selected:
        clip = clips[row["video_id"]]
        path = MEDIA / f"{row['video_id']}.mp4"
        if digest(path) != clip["sha256"]:
            raise ValueError("media hash mismatch")
        row["media_path"] = str(path)
        row["media_sha256"] = clip["sha256"]
    result = {
        "status": "PASS_SAMPLE_FROZEN",
        "population": len(entries),
        "population_by_stratum": dict(Counter(r["category"] for r in entries)),
        "samples": selected,
        "manifest_sha256": MANIFEST_SHA,
        "labels_sha256": PINNED[LABELS],
        "archive_sha256": ARCHIVE_SHA,
        "builder_sha256": BUILDER_SHA,
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": protocol_sha,
        "scope": "stratified train-source diagnostic; no independent media judgments yet",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
