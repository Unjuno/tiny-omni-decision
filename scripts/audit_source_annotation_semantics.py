"""Audit source-label interpretation against pinned official train annotations."""

import argparse
import hashlib
import importlib.util
import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.intervene_source_same_label_audio import ARCHIVE, ARCHIVE_SHA
from scripts.train_source_unimodal_cv import LABELS, PINNED, SOURCE, digest

BUILDER = SOURCE / "src/tiny_omni_decision/perception_source_candidate.py"
BUILDER_SHA = "78263a14fddfdc06d50951207ea0b72e999e905b97399bcc08171de92524dc3f"


def audit_record(record, source_video, builder):
    """Reconstruct candidate/object identities and audit related-parent presence."""
    video = record["group_id"]
    if record["split"] != "train" or record["media_reference"]["video_id"] != video:
        raise ValueError("invalid record split or media identity")
    sound_id = record["provenance"]["sound_annotation_id"]
    matching = [s for s in source_video["sound_localisation"] if s["id"] == sound_id]
    if len(matching) != 1:
        raise ValueError("missing or duplicate sound annotation")
    sound = matching[0]
    start, end = sound["frame_ids"]
    cut = record["provenance"]["upstream_cut_frame"]
    objects = []
    for obj in source_video["object_tracking"]:
        frames, boxes = obj["frame_ids"], obj["bounding_boxes"]
        flags = obj.get("is_masked")
        if flags is None:
            flags = [0] * len(frames)
        if len(frames) != len(boxes) or len(frames) != len(flags):
            raise ValueError("unaligned track")
        selected = [
            {"frame_id": frame, "bbox_xyxy_normalized": builder._valid_box(box)}
            for frame, box, masked in zip(frames, boxes, flags, strict=True)
            if start <= frame <= end and (cut < 0 or frame < cut) and masked == 0
        ]
        if selected:
            key = hashlib.sha256(f"{video}:{sound_id}:{obj['id']}".encode()).hexdigest()
            objects.append((key, obj["id"], builder._sample_frames(selected)))
    objects.sort()
    candidates = record["request"]["candidates"]
    if len(candidates) != len(objects):
        raise ValueError("candidate count mismatch")
    for index, (candidate, obj) in enumerate(zip(candidates, objects, strict=True)):
        if candidate["candidate_id"] != f"option_{index}" or candidate["frame_boxes"] != obj[2]:
            raise ValueError("candidate reconstruction mismatch")
    parents = set(sound["parent_objects"]) - {-1}
    matched = {f"option_{i}" for i, (_, obj_id, _) in enumerate(objects) if obj_id in parents}
    is_none = record["supervision"]["none_visible"]
    if is_none != (sound["is_visible"] == 0):
        raise ValueError("visibility mapping mismatch")
    if not is_none and matched != set(record["supervision"]["target_candidate_ids"]):
        raise ValueError("visible parent target mismatch")
    return {
        "none": is_none,
        "has_unannotated_parent": -1 in sound["parent_objects"],
        "has_known_parent": bool(parents),
        "known_parent_candidates": len(matched),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path, expected in (
        (ARCHIVE, ARCHIVE_SHA),
        (LABELS, PINNED[LABELS]),
        (BUILDER, BUILDER_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location("source_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    records = list(map(json.loads, LABELS.read_bytes().splitlines()))
    with zipfile.ZipFile(ARCHIVE) as archive:
        source = json.loads(archive.read("all_train.json"))
    counts = Counter()
    for record in records:
        item = audit_record(record, source[record["group_id"]], builder)
        counts["records"] += 1
        if item["none"]:
            counts["none"] += 1
            counts["none_unannotated_parent"] += item["has_unannotated_parent"]
            counts["none_known_parent"] += item["has_known_parent"]
            counts["none_known_parent_is_candidate"] += item["known_parent_candidates"] > 0
            counts["none_known_parent_candidate_count"] += item["known_parent_candidates"]
        else:
            counts["visible"] += 1
            counts["visible_multiple_parents"] += item["known_parent_candidates"] > 1
    if counts["records"] != 2593:
        raise ValueError("unexpected record count")
    result = {
        "status": "PASS_SOURCE_LABEL_RECONSTRUCTION",
        "counts": dict(counts),
        "archive_sha256": ARCHIVE_SHA,
        "labels_sha256": PINNED[LABELS],
        "builder_sha256": BUILDER_SHA,
        "script_sha256": digest(Path(__file__)),
        "scope": "previously inspected train-source annotations only; no media or audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
