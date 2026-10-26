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
QUESTION = "Which audio option was recorded with this video?"
OPTIONS = ("Audio A", "Audio B")


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


def make_pairs(assignments):
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
                    correct_audio = video_record
                    distractor_audio = action_groups[action][
                        (index + 1) % balanced_count
                    ]
                    video_group = video_record["group_id"]
                    correct_group = correct_audio["group_id"]
                    distractor_group = distractor_audio["group_id"]
                    if correct_group != video_group or distractor_group == video_group:
                        raise ValueError("candidate audio source identity is inconsistent")
                    used_groups[split].update(
                        (video_group, correct_group, distractor_group)
                    )
                    identity = f"{split}|Ar{block[0]}|V{block[1]}|{action}|{index}"
                    for order_index in (0, 1):
                        row_id = stable_id(f"{identity}|order{order_index}")
                        ordered_audio = (
                            [correct_audio, distractor_audio]
                            if order_index == 0
                            else [distractor_audio, correct_audio]
                        )
                        manifests[split].append(
                            {
                                "id": row_id,
                                "model_input": {
                                    "question": QUESTION,
                                    "options": list(OPTIONS),
                                },
                                "target": order_index,
                                "source_refs": {
                                    "video": {
                                        "group_id": video_group,
                                        "archive": video_record["archive"],
                                        "member": input_member(
                                            video_record, "_Rx1_cam.mp4"
                                        ),
                                    },
                                    "audio_candidates": [
                                        {
                                            "group_id": record["group_id"],
                                            "archive": record["archive"],
                                            "member": input_member(
                                                record, "_Rx1_mic.wav"
                                            ),
                                        }
                                        for record in ordered_audio
                                    ],
                                },
                            }
                        )
        rows = manifests[split]
        target_positions = Counter(row["target"] for row in rows)
        if not rows or target_positions[0] != target_positions[1]:
            raise ValueError("each split must balance the correct option position")
        video_position_counts = Counter()
        audio_role_counts = Counter()
        for row in rows:
            video_group = row["source_refs"]["video"]["group_id"]
            video_position_counts[(video_group, row["target"])] += 1
            for index, candidate in enumerate(row["source_refs"]["audio_candidates"]):
                role = "correct" if index == row["target"] else "distractor"
                audio_role_counts[(candidate["group_id"], role)] += 1
        if any(
            video_position_counts[(gid, 0)] != video_position_counts[(gid, 1)]
            for gid in used_groups[split]
        ):
            raise ValueError("video assets are not balanced across correct option positions")
        if any(
            audio_role_counts[(gid, "correct")] != audio_role_counts[(gid, "distractor")]
            for gid in used_groups[split]
        ):
            raise ValueError("audio assets are not balanced as correct and distractor")

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
    manifests, accounting = make_pairs(assignments)
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
        "correct_option_position_counts": {
            split: dict(Counter(str(row["target"]) for row in rows))
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
