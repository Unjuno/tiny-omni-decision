"""Reject an underspecified task before fitting it to inconsistent supervision."""

import argparse
import collections
import json
import zipfile
from pathlib import Path

from scripts.audit_perception_train import sha


def identity_audit(data, include_sound_class=False):
    groups = collections.defaultdict(list)
    for video_id, row in data.items():
        if row["metadata"]["split"] != "train":
            raise ValueError("train-only audit")
        for sound in row["sound_localisation"]:
            key = (video_id, *sound["timestamps"])
            if include_sound_class:
                key += (sound["label"],)
            groups[key].append(sound)
    conflicts = [
        dict(
            input_key=list(key),
            annotations=[
                {k: s[k] for k in ("id", "label", "timestamps", "is_visible")} for s in rows
            ],
        )
        for key, rows in groups.items()
        if len({s["is_visible"] for s in rows}) > 1
    ]
    return dict(
        unique_inputs=len(groups),
        contradictory_inputs=len(conflicts),
        contradictory_annotations=sum(len(g["annotations"]) for g in conflicts),
        conflicts=conflicts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    if sha(args.source) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("source changed")
    with zipfile.ZipFile(args.source) as z:
        data = json.loads(z.read("all_train.json"))
    report = dict(
        source_sha256=sha(args.source),
        interval_only=identity_audit(data),
        class_conditioned=identity_audit(data, True),
        training=False,
        evaluation_opened=False,
        limitation="Exact collisions only; no proof that all source semantics are unambiguous.",
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {a: b for a, b in report[k].items() if a != "conflicts"}
                for k in ("interval_only", "class_conditioned")
            }
        )
    )


if __name__ == "__main__":
    main()
