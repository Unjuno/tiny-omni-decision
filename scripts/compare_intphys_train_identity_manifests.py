"""Compare train-shard source IDs, canonical statuses, and full RGB identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_identities(shard_number: int, paths: list[Path]) -> list[dict[str, str]]:
    """Load hashed identities from train/validation frozen-feature archives."""
    if shard_number < 1 or not paths:
        raise ValueError("shard number and at least one feature archive are required")
    records: list[dict[str, str]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"sample_ids", "status_identity_sha256", "rgb_sequence_sha256"}
            if not required.issubset(data.files):
                raise ValueError(f"feature archive lacks identity columns: {path}")
            columns = [
                data[name].astype(str).tolist()
                for name in (
                    "sample_ids",
                    "status_identity_sha256",
                    "rgb_sequence_sha256",
                )
            ]
        if len({len(column) for column in columns}) != 1:
            raise ValueError(f"feature archive identity columns differ in length: {path}")
        records.extend(
            {
                "sample_id": f"train.{shard_number}/{sample_id}",
                "source_id": sample_id,
                "status_identity_sha256": status_hash,
                "rgb_sequence_sha256": rgb_hash,
            }
            for sample_id, status_hash, rgb_hash in zip(*columns, strict=True)
        )
    return records


def load_identity_manifest(path: Path, *, shard_number: int) -> list[dict[str, str]]:
    """Load a JSONL manifest created by the pinned archive-structure audit."""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        sample_id = str(raw.get("sample_id", ""))
        prefix = f"train.{shard_number}/"
        if not sample_id.startswith(prefix):
            raise ValueError(f"identity manifest has unexpected shard ID at line {line_number}")
        records.append(
            {
                "sample_id": sample_id,
                "source_id": sample_id.removeprefix(prefix),
                "status_identity_sha256": str(raw.get("status_identity_sha256", "")),
                "rgb_sequence_sha256": str(raw.get("rgb_sequence_sha256", "")),
            }
        )
    return records


def compare_identity_records(
    shard_records: dict[int, list[dict[str, str]]],
) -> dict[str, Any]:
    """Count exact cross-shard overlaps and union records by any underlying identity."""
    if len(shard_records) < 2 or any(
        shard < 1 or not rows for shard, rows in shard_records.items()
    ):
        raise ValueError("at least two nonempty train-shard identity lists are required")
    ordered_shards = sorted(shard_records)
    all_records = []
    source_ids_by_shard: dict[int, set[str]] = {}
    statuses_by_shard: dict[int, set[str]] = {}
    rgb_by_shard: dict[int, set[str]] = {}
    for shard in ordered_shards:
        rows = shard_records[shard]
        sample_ids = [row["sample_id"] for row in rows]
        if len(set(sample_ids)) != len(sample_ids) or any(
            not sample_id.startswith(f"train.{shard}/") for sample_id in sample_ids
        ):
            raise ValueError(f"shard {shard} has duplicate or incorrectly qualified sample IDs")
        for row in rows:
            for key in ("source_id", "status_identity_sha256", "rgb_sequence_sha256"):
                if not row.get(key):
                    raise ValueError(f"shard {shard} identity row lacks {key}")
            for key in ("status_identity_sha256", "rgb_sequence_sha256"):
                value = row[key]
                if len(value) != 64:
                    raise ValueError(f"shard {shard} has malformed {key}")
                try:
                    bytes.fromhex(value)
                except ValueError as error:
                    raise ValueError(f"shard {shard} has malformed {key}") from error
        all_records.extend(rows)
        source_ids_by_shard[shard] = {row["source_id"] for row in rows}
        statuses_by_shard[shard] = {row["status_identity_sha256"] for row in rows}
        rgb_by_shard[shard] = {row["rgb_sequence_sha256"] for row in rows}

    parent = list(range(len(all_records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(all_records):
        for kind in ("source_id", "status_identity_sha256", "rgb_sequence_sha256"):
            key = (kind, row[kind])
            previous = owners.setdefault(key, index)
            union(index, previous)

    cross_pairs = [
        (left, right)
        for position, left in enumerate(ordered_shards)
        for right in ordered_shards[position + 1 :]
    ]
    return {
        "status": "cross_shard_train_identity_comparison",
        "source_shards": ordered_shards,
        "sample_counts_by_shard": {
            str(shard): len(shard_records[shard]) for shard in ordered_shards
        },
        "unqualified_source_id_overlap_by_shard_pair": {
            f"{left}-{right}": len(source_ids_by_shard[left] & source_ids_by_shard[right])
            for left, right in cross_pairs
        },
        "exact_status_identity_overlap_by_shard_pair": {
            f"{left}-{right}": len(statuses_by_shard[left] & statuses_by_shard[right])
            for left, right in cross_pairs
        },
        "exact_full_rgb_sequence_overlap_by_shard_pair": {
            f"{left}-{right}": len(rgb_by_shard[left] & rgb_by_shard[right])
            for left, right in cross_pairs
        },
        "global_unique_groups_after_identity_union": len(
            {find(index) for index in range(len(all_records))}
        ),
        "records_removed_by_global_deduplication": len(all_records)
        - len({find(index) for index in range(len(all_records))}),
        "identity_kinds_unioned": [
            "unqualified source ID",
            "canonical status",
            "full ordered RGB sequence",
        ],
        "normalized_sampled_input_identity_compared": False,
        "validation_or_sealed_data_accessed": False,
        "model_or_gpu_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard1-train", type=Path, required=True)
    parser.add_argument("--shard1-validation", type=Path, required=True)
    parser.add_argument("--shard2-identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shard1_paths = [
        args.shard1_train.resolve(strict=True),
        args.shard1_validation.resolve(strict=True),
    ]
    shard2_path = args.shard2_identity_manifest.resolve(strict=True)
    result = compare_identity_records(
        {
            1: load_feature_identities(1, shard1_paths),
            2: load_identity_manifest(shard2_path, shard_number=2),
        }
    )
    result["input_sha256"] = {
        "shard1_train_features": sha256_file(shard1_paths[0]),
        "shard1_validation_features": sha256_file(shard1_paths[1]),
        "shard2_identity_manifest": sha256_file(shard2_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["receipt_sha256"] = sha256_file(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
