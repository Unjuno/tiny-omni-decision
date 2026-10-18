"""Build a leak-resistant, training-only audio/video correspondence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 17
MAX_MEDIA_MEMBER_BYTES = 40 * 1024 * 1024
MAX_COMPRESSED_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_COMPRESSED_BYTES = 1 * 1024 * 1024 * 1024
EXPECTED_PARTITION_SHA256 = "6c9ac8eb8e12bd5dc0af77c8d162a40c753d5a215d07eb85dc3a69c3973d9f98"
EXPECTED_ANNOTATIONS_SHA256 = "24e8df777bee34102591e7b15901b931034ee2df3b549d632fb5b7094322e3dd"
EXPECTED_SOURCE_REVIEW_PLAN_SHA256 = (
    "951d1078d3fae228256bb398bd636fea2393fd7aa4b05969ef04655c7f475ae5"
)
EXPECTED_ARCHIVE_INDEX_SHA256 = "98673a0101020606537d2850f825ebe08219c496928002f2cf09dcaa83b9824a"
QUESTION = "Do the audio and video show the same recorded material event?"
OPTIONS = ["No, they are different recordings.", "Yes, they are the same recording."]
SPLIT_COUNTS = (0.6, 0.2, 0.2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_id(parent_id: str) -> str:
    """Remove a seventh numeric clip suffix, preserving the six-part timestamp."""
    parts = parent_id.split("-")
    if len(parts) == 7 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    return parent_id


def _stable_key(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode()).digest()


def _dominant_signature(rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    counts = Counter(
        (str(fields[0]), str(fields[1]))
        for row in rows
        if isinstance((fields := row.get("fields")), list)
        and len(fields) >= 2
        and fields[0] is not None
        and fields[1] is not None
    )
    if not counts:
        return None
    highest = max(counts.values())
    return min(signature for signature, count in counts.items() if count == highest)


def _split_signatures(
    groups_by_signature: dict[tuple[str, str], list[str]], seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    assignments: dict[str, str] = {}
    accounting: dict[str, Any] = {}
    for signature, groups in sorted(groups_by_signature.items()):
        eligible = sorted(groups, key=lambda value: _stable_key(seed, "class-order", value))
        original_count = len(eligible)
        if original_count < 10:
            accounting["|".join(signature)] = {
                "eligible_groups": original_count,
                "excluded_groups": original_count,
                "reason": "fewer_than_10_same-signature_groups",
            }
            continue
        odd_exclusion = eligible.pop() if len(eligible) % 2 else None
        count = len(eligible)
        train_count = int((count * SPLIT_COUNTS[0]) // 2) * 2
        calibration_count = int((count * SPLIT_COUNTS[1]) // 2) * 2
        validation_count = count - train_count - calibration_count
        if min(train_count, calibration_count, validation_count) < 2 or validation_count % 2:
            raise ValueError(f"cannot form even >=2 source groups per split for {signature}")
        boundaries = {
            "train": train_count,
            "calibration": train_count + calibration_count,
            "validation": count,
        }
        for index, group_id in enumerate(eligible):
            split = (
                "train"
                if index < boundaries["train"]
                else ("calibration" if index < boundaries["calibration"] else "validation")
            )
            assignments[group_id] = split
        accounting["|".join(signature)] = {
            "eligible_groups": count,
            "odd_group_excluded_for_balanced_pairing": odd_exclusion,
            "split_groups": {
                "train": train_count,
                "calibration": calibration_count,
                "validation": validation_count,
            },
        }
    return assignments, accounting


def _make_rows(
    representatives: dict[str, str],
    signatures: dict[str, tuple[str, str]],
    assignments: dict[str, str],
    archive_members: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_split_signature: dict[tuple[str, tuple[str, str]], list[str]] = defaultdict(list)
    for group_id, split in assignments.items():
        by_split_signature[(split, signatures[group_id])].append(group_id)

    output: dict[str, list[dict[str, Any]]] = {"train": [], "calibration": [], "validation": []}
    for (split, signature), group_ids in sorted(by_split_signature.items()):
        ordered = sorted(group_ids, key=lambda value: _stable_key(seed, "pair-order", value))
        if len(ordered) % 2:
            raise ValueError("split/class source groups must be even for marginal balance")
        for pair_index in range(0, len(ordered), 2):
            left, right = ordered[pair_index : pair_index + 2]
            pair_id = hashlib.sha256(f"{split}\0{signature}\0{left}\0{right}".encode()).hexdigest()[
                :20
            ]
            pairs = (
                (left, left, 1),
                (right, right, 1),
                (left, right, 0),
                (right, left, 0),
            )
            for row_index, (audio_group, video_group, target) in enumerate(pairs):
                audio_parent = representatives[audio_group]
                video_parent = representatives[video_group]
                audio_name = f"{audio_parent}_denoised.wav"
                video_name = f"{video_parent}_denoised.mp4"
                if audio_name not in archive_members or video_name not in archive_members:
                    raise ValueError(
                        f"source archive index is missing media for {audio_parent}/{video_parent}"
                    )
                output[split].append(
                    {
                        "id": f"gh-av-match-{split}-{pair_id}-{row_index}",
                        "source": "Greatest-Hits@official-CC-BY-4.0",
                        "source_signature": list(signature),
                        "question": QUESTION,
                        "options": OPTIONS,
                        "target": target,
                        "audio_path": audio_name,
                        "video_path": video_name,
                        "audio_source_group": audio_group,
                        "video_source_group": video_group,
                        "pair_cluster_id": pair_id,
                        "pair_index": row_index,
                    }
                )
    for rows in output.values():
        rows.sort(key=lambda row: row["id"])
    return output


def build_probe(
    partition: dict[str, list[str]],
    annotations: dict[str, list[dict[str, Any]]],
    reviewed_plan: list[dict[str, Any]],
    archive_index: list[dict[str, Any]],
    seed: int = SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build a train-only binary AV match task with exactly balanced marginals."""
    train_ids = set(partition.get("train", []))
    if not train_ids:
        raise ValueError("the pinned source partition has no training IDs")
    reviewed_groups = {_group_id(str(row["parent"])) for row in reviewed_plan}
    all_partition_ids = set().union(*(set(values) for values in partition.values()))
    if len(all_partition_ids) != sum(len(values) for values in partition.values()):
        raise ValueError("source IDs overlap partition splits")

    grouped: dict[str, list[str]] = defaultdict(list)
    for parent_id in sorted(train_ids):
        grouped[_group_id(parent_id)].append(parent_id)
    excluded_reviewed = set(grouped) & reviewed_groups
    grouped = {
        group_id: ids for group_id, ids in grouped.items() if group_id not in reviewed_groups
    }

    representatives = {group_id: sorted(ids)[0] for group_id, ids in grouped.items()}
    signatures: dict[str, tuple[str, str]] = {}
    for group_id, parent_id in representatives.items():
        signature = _dominant_signature(annotations.get(parent_id, []))
        if signature is not None:
            signatures[group_id] = signature
    groups_by_signature: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, signature in signatures.items():
        groups_by_signature[signature].append(group_id)
    assignments, class_accounting = _split_signatures(groups_by_signature, seed)

    member_by_name: dict[str, dict[str, Any]] = {}
    for member in archive_index:
        name = str(member.get("name", "")).rstrip("/")
        if name:
            member_by_name[Path(name).name] = member
    rows = _make_rows(representatives, signatures, assignments, member_by_name, seed)

    selected_media = {
        path: member_by_name[path]
        for split_rows in rows.values()
        for row in split_rows
        for path in (row["audio_path"], row["video_path"])
    }
    expanded_bytes = sum(int(member["bytes"]) for member in selected_media.values())
    compressed_bytes = sum(int(member["compressed"]) for member in selected_media.values())
    if any(
        int(member["bytes"]) > MAX_MEDIA_MEMBER_BYTES
        or int(member["compressed"]) > MAX_COMPRESSED_MEMBER_BYTES
        for member in selected_media.values()
    ):
        raise ValueError("selected media member exceeds its per-member transfer/decode bound")
    if expanded_bytes > MAX_TOTAL_MEDIA_BYTES or compressed_bytes > MAX_TOTAL_COMPRESSED_BYTES:
        raise ValueError("selected media exceeds its aggregate transfer/decode bound")

    split_groups = {
        split: sorted(group_id for group_id, value in assignments.items() if value == split)
        for split in ("train", "calibration", "validation")
    }
    if set(split_groups["train"]) & set(split_groups["calibration"]):
        raise ValueError("group leakage between train and calibration")
    if (set(split_groups["train"]) | set(split_groups["calibration"])) & set(
        split_groups["validation"]
    ):
        raise ValueError("group leakage into validation")

    # A fixed two-source cluster has one positive and one negative occurrence
    # for each audio source and each video source. A unimodal lookup has exactly
    # balanced targets by construction.
    modality_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    pair_targets: dict[str, list[int]] = defaultdict(list)
    for split_rows in rows.values():
        for row in split_rows:
            cluster = row["pair_cluster_id"]
            pair_targets[cluster].append(int(row["target"]))
            modality_targets[("audio", row["audio_source_group"])].add(int(row["target"]))
            modality_targets[("video", row["video_source_group"])].add(int(row["target"]))
    if any(sorted(targets) != [0, 0, 1, 1] for targets in pair_targets.values()):
        raise ValueError("a pair cluster is not balanced across match and mismatch")
    if any(targets != {0, 1} for targets in modality_targets.values()):
        raise ValueError("unimodal target marginals are not balanced")

    summary = {
        "status": "preflight_only_no_media_or_weights_loaded",
        "experiment_id": "greatest-hits-av-match-v0",
        "seed": seed,
        "source_split_used": "train only",
        "source_parent_count": len(train_ids),
        "reviewed_source_group_exclusions": sorted(excluded_reviewed),
        "remaining_source_groups": len(grouped),
        "signature_groups": len(signatures),
        "selected_groups": {split: len(ids) for split, ids in split_groups.items()},
        "pair_clusters": {
            split: len({row["pair_cluster_id"] for row in split_rows})
            for split, split_rows in rows.items()
        },
        "rows": {split: len(split_rows) for split, split_rows in rows.items()},
        "selected_media_files": len(selected_media),
        "selected_media_expanded_bytes": expanded_bytes,
        "selected_media_compressed_bytes": compressed_bytes,
        "media_member_limits": {
            "expanded_bytes": MAX_MEDIA_MEMBER_BYTES,
            "compressed_bytes": MAX_COMPRESSED_MEMBER_BYTES,
            "aggregate_expanded_bytes": MAX_TOTAL_MEDIA_BYTES,
            "aggregate_compressed_bytes": MAX_TOTAL_COMPRESSED_BYTES,
        },
        "class_accounting": class_accounting,
        "split_source_group_ids": split_groups,
        "sample_id_order_sha256": {
            split: hashlib.sha256("\n".join(row["id"] for row in split_rows).encode()).hexdigest()
            for split, split_rows in rows.items()
        },
        "group_overlap": 0,
        "reviewed_source_group_overlap": 0,
        "pair_marginals_balanced": True,
        "underlying_media_files_opened": False,
        "sealed_audit_opened": False,
        "optimizer_updates": 0,
    }
    return rows, summary


def _load_pinned_json(path: Path, expected_sha256: str) -> Any:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"pinned input hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--source-review-plan", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")

    partition_path = args.inventory_dir / "partition.json"
    annotations_path = args.inventory_dir / "annotations.json"
    partition = _load_pinned_json(partition_path, EXPECTED_PARTITION_SHA256)
    annotations = _load_pinned_json(annotations_path, EXPECTED_ANNOTATIONS_SHA256)
    reviewed_plan = _load_pinned_json(args.source_review_plan, EXPECTED_SOURCE_REVIEW_PLAN_SHA256)
    archive_index = _load_pinned_json(args.archive_index, EXPECTED_ARCHIVE_INDEX_SHA256)
    rows, summary = build_probe(partition, annotations, reviewed_plan, archive_index)
    summary["input_sha256"] = {
        "partition.json": EXPECTED_PARTITION_SHA256,
        "annotations.json": EXPECTED_ANNOTATIONS_SHA256,
        "greatest-hits-source-review-plan.json": EXPECTED_SOURCE_REVIEW_PLAN_SHA256,
        "official-archive-directory-index.json": EXPECTED_ARCHIVE_INDEX_SHA256,
    }

    args.output_dir.mkdir(parents=True)
    for split, split_rows in rows.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows), encoding="utf-8"
        )
    (args.output_dir / "preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["output_sha256"] = {
        file.name: _sha256(file) for file in sorted(args.output_dir.iterdir()) if file.is_file()
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "selected_groups",
                    "pair_clusters",
                    "rows",
                    "group_overlap",
                    "pair_marginals_balanced",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
