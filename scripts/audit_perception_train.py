"""Train-only source/task feasibility; never imports models or opens evaluation data."""

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(data, cuts):
    counts = collections.Counter()
    tags, visibility, labels = collections.Counter(), collections.Counter(), collections.Counter()
    for video_id, row in data.items():
        metadata = row["metadata"]
        if metadata["split"] != "train" or metadata["video_id"] != video_id:
            raise ValueError("train-only identity mismatch")
        counts["videos"] += 1
        if video_id not in cuts:
            counts["missing_cut_mapping"] += 1
        elif cuts[video_id] >= 0:
            counts["cut_required_videos"] += 1
            if not 0 <= cuts[video_id] < metadata["num_frames"]:
                counts["invalid_cut_frame"] += 1
        for q in row["mc_question"]:
            counts["mc_questions"] += 1
            if len(q["options"]) != 3 or q["answer_id"] not in range(3):
                raise ValueError("unexpected MC target/options")
            tags.update(q["tag"])
        objects = {o["id"] for o in row["object_tracking"]}
        if row["sound_localisation"]:
            counts["videos_with_sound_annotations"] += 1
        usable = False
        for sound in row["sound_localisation"]:
            counts["sound_segments"] += 1
            visibility[str(sound["is_visible"])] += 1
            labels[sound["label"]] += 1
            parents = sound["parent_objects"]
            known = bool(parents) and all(p in objects for p in parents)
            counts["sound_segments_with_known_parents"] += int(known)
            if sound["is_visible"] not in (0, 1):
                raise ValueError("unknown visibility semantics")
            start, end = sound["timestamps"]
            if start < 0 or end <= start:
                counts["invalid_sound_time"] += 1
            usable |= known
        counts["videos_with_known_sound_parent"] += int(usable)
    return dict(counts=counts, mc_tags=tags, sound_visibility=visibility, sound_labels=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    archive = args.source / "train_annotations.zip"
    if sha(archive) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("unreviewed archive revision")
    cuts_path = args.source / "cut_frame_mapping_train.json"
    with zipfile.ZipFile(archive) as z:
        if (
            z.namelist() != ["all_train.json"]
            or z.getinfo("all_train.json").file_size > 150_000_000
        ):
            raise ValueError("unexpected archive content")
        data = json.loads(z.read("all_train.json"))
    report = summarize(data, json.loads(cuts_path.read_text(encoding="utf-8")))
    report.update(
        source_revision="3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
        hashes={p.name: sha(p) for p in (archive, cuts_path, args.source / "README.md")},
        validation_opened=False,
        test_opened=False,
        media_downloaded=False,
        admission="REVIEW: task and contributor grouping unresolved",
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]))
    print(json.dumps(report["sound_visibility"]))


if __name__ == "__main__":
    main()
