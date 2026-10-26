"""Build deterministic group-disjoint audio/video action-correspondence manifests."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE = re.compile(
    r"(?P<index>\d{2})_Ar(?P<robot>[12])_V(?P<velocity>high|med|low)_Ac(?P<action>[0-9])\.zip"
)
SPLITS = ("train", "validation", "sealed_audit")
QUESTION = "Do these audio and video clips show the same action category?"
OPTIONS = ("different action", "same action")


def stable_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def split_groups(directory_audit, seed=17):
    if directory_audit["status"] != "PASS_DIRECTORY_METADATA_ONLY":
        raise ValueError("verified directory-only inventory required")
    assignments = {split: {} for split in SPLITS}
    records = {}
    for archive in directory_audit["archives"]:
        match = ARCHIVE.fullmatch(archive["archive"])
        if not match:
            raise ValueError("unknown source archive class")
        labels = match.groupdict()
        labels["action"] = int(labels["action"])
        groups = list(archive["groups"])
        if len(groups) != archive["repetition_groups"] or len(groups) < 16:
            raise ValueError("archive repetition inventory is incomplete or too small")
        rng = random.Random(seed + int(labels["index"]))
        rng.shuffle(groups)
        held_out = max(2, len(groups) // 8)
        if 2 * held_out >= len(groups):
            raise ValueError("not enough repetitions for train/validation/audit")
        partitions = {
            "train": groups[: len(groups) - 2 * held_out],
            "validation": groups[len(groups) - 2 * held_out : len(groups) - held_out],
            "sealed_audit": groups[len(groups) - held_out :],
        }
        for split, split_groups_ in partitions.items():
            for group in split_groups_:
                gid = group["group_id"]
                if any(gid in assignments[other] for other in SPLITS):
                    raise ValueError("one source repetition assigned to multiple splits")
                assignments[split][gid] = {
                    **labels,
                    "archive": archive["archive"],
                    "group_id": gid,
                    "input_members": group["input_members"],
                    "supervision_only_member": group["supervision_only_member"],
                }
                records[gid] = assignments[split][gid]
    return assignments, records


def input_member(record, suffix):
    matches = [
        name
        for name in record["input_members"]
        if name.endswith(suffix) and "_Rx1_" in name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one fixed Rx1 input ending in {suffix}")
    return matches[0]


def make_pairs(assignments, seed=17):
    manifests = {split: [] for split in SPLITS}
    used_groups = {split: set() for split in SPLITS}
    dropped = Counter()
    for split in SPLITS:
        by_block_action = defaultdict(list)
        for record in assignments[split].values():
            block = (record["robot"], record["velocity"])
            by_block_action[(block, record["action"])].append(record)
        for block in sorted({key[0] for key in by_block_action}):
            action_groups = {
                action: sorted(
                    by_block_action[(block, action)], key=lambda record: record["group_id"]
                )
                for action in range(10)
            }
            if any(len(action_groups[action]) < 2 for action in range(10)):
                raise ValueError("each robot/velocity block needs two groups per action")
            balanced_count = min(len(groups) for groups in action_groups.values())
            for action, groups in action_groups.items():
                dropped[f"{split}:Ar{block[0]}:V{block[1]}"] += len(groups) - balanced_count
                action_groups[action] = groups[:balanced_count]
            for action in range(10):
                for index, video_record in enumerate(action_groups[action]):
                    same_action_audio = action_groups[action][(index + 1) % balanced_count]
                    different_action_audio = action_groups[(action + 1) % 10][index]
                    for label, audio_record in (
                        ("same action", same_action_audio),
                        ("different action", different_action_audio),
                    ):
                        identity = (
                            f"{split}|Ar{block[0]}|V{block[1]}|{action}|{index}|{label}"
                        )
                        row_id = stable_id(identity)
                        options = list(OPTIONS)
                        random.Random(f"{seed}:{row_id}").shuffle(options)
                        target = options.index(
                            label
                        )
                        video_group = video_record["group_id"]
                        audio_group = audio_record["group_id"]
                        if video_group == audio_group:
                            raise ValueError("paired task must use separate recording groups")
                        used_groups[split].update((video_group, audio_group))
                        manifests[split].append(
                            {
                                "id": row_id,
                                "model_input": {
                                    "question": QUESTION,
                                    "options": options,
                                },
                                "target": target,
                                "source_refs": {
                                    "video": {
                                        "group_id": video_group,
                                        "archive": video_record["archive"],
                                        "member": input_member(
                                            video_record, "_Rx1_cam.mp4"
                                        ),
                                    },
                                    "audio": {
                                        "group_id": audio_group,
                                        "archive": audio_record["archive"],
                                        "member": input_member(
                                            audio_record, "_Rx1_mic.wav"
                                        ),
                                    },
                                },
                            }
                        )
        rows = manifests[split]
        positives = sum(
            row["model_input"]["options"][row["target"]] == "same action" for row in rows
        )
        if not rows or positives * 2 != len(rows):
            raise ValueError("each split must have equal same/different task labels")
        video_label_counts = Counter()
        audio_label_counts = Counter()
        for row in rows:
            label = row["model_input"]["options"][row["target"]]
            video_label_counts[(row["source_refs"]["video"]["group_id"], label)] += 1
            audio_label_counts[(row["source_refs"]["audio"]["group_id"], label)] += 1
        if any(
            video_label_counts[(gid, "same action")]
            != video_label_counts[(gid, "different action")]
            for gid in used_groups[split]
        ):
            raise ValueError("video assets are not label-balanced")
        if any(
            audio_label_counts[(gid, "same action")]
            != audio_label_counts[(gid, "different action")]
            for gid in used_groups[split]
        ):
            raise ValueError("audio assets are not label-balanced")

    split_pairs = (
        ("train", "validation"),
        ("train", "sealed_audit"),
        ("validation", "sealed_audit"),
    )
    for left, right in split_pairs:
        if used_groups[left] & used_groups[right]:
            raise ValueError("source repetition overlaps across splits")
    return manifests, {
        "used_groups_by_split": {split: len(groups) for split, groups in used_groups.items()},
        "dropped_groups_by_block": dict(dropped),
    }


def build(directory_audit, seed=17):
    assignments, _ = split_groups(directory_audit, seed)
    manifests, accounting = make_pairs(assignments, seed)
    return manifests, accounting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sealed-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output_dir.exists() or args.sealed_output_dir.exists():
        raise ValueError("choose fresh train/validation and sealed output directories")
    if args.output_dir.resolve() == args.sealed_output_dir.resolve():
        raise ValueError("sealed-audit output must be separate from train/validation")
    train_path = args.output_dir.resolve()
    audit_path = args.sealed_output_dir.resolve()
    if train_path in audit_path.parents or audit_path in train_path.parents:
        raise ValueError("train/validation and sealed output directories must be disjoint")
    directory_audit = json.loads(args.directory_audit.read_bytes())
    manifests, accounting = build(directory_audit, args.seed)
    args.output_dir.mkdir(parents=True)
    args.sealed_output_dir.mkdir(parents=True)
    summary = {
        "status": "PASS_METADATA_ONLY_GROUP_DISJOINT_PAIRED_TASK_MANIFESTS",
        "seed": args.seed,
        "directory_audit_sha256": hashlib.sha256(args.directory_audit.read_bytes()).hexdigest(),
        "question": QUESTION,
        "options": list(OPTIONS),
        "split_row_counts": {split: len(rows) for split, rows in manifests.items()},
        "split_pair_counts": {
            split: dict(
                Counter(row["model_input"]["options"][row["target"]] for row in rows)
            )
            for split, rows in manifests.items()
        },
        "accounting": accounting,
        "input_group_overlap": False,
        "payload_opened": False,
        "training": False,
        "model_inference": False,
        "sealed_audit_evaluated": False,
        "limitations": [
            "filename-derived repetition groups do not prove independent collection sessions",
            "metadata only; no quality, unimodal or timing result is claimed",
            "sealed-audit manifest is not loaded by the training or validation path",
        ],
    }
    for split, rows in manifests.items():
        base = args.sealed_output_dir if split == "sealed_audit" else args.output_dir
        path = base / f"{split}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
            stream.write("\n")
        summary[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print("PASS_METADATA_ONLY_GROUP_DISJOINT_PAIRED_TASK_MANIFESTS", summary["split_row_counts"])


if __name__ == "__main__":
    main()
