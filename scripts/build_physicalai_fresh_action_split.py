"""Make a fresh episode-held development split from unused source-train groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PINNED_SOURCE_MANIFEST_SHA256 = (
    "3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rank_key(row: dict[str, Any], *, seed: int, protocol_id: str) -> str:
    source = row["episode_index"]
    return _sha256(
        f"{protocol_id}|seed={seed}|episode_index={source:06d}".encode()
    )


def _id_order_hash(rows: list[dict[str, Any]]) -> str:
    return _sha256("\n".join(f"{row['episode_index']:06d}" for row in rows).encode())


def _split_hash(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        f"{row['episode_index']:06d}\t{row['video_sha256']}\t"
        f"{row['parquet_sha256']}\n"
        for row in sorted(rows, key=lambda item: item["episode_index"])
    )
    return _sha256(body.encode())


def validate_fresh_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on leakage, missing assets, or an exposed reserve cohort."""
    expected_id = "nvidia-physicalai-singlearm-stack-platforms-fresh-action-v1"
    if manifest.get("candidate_id") != expected_id:
        raise ValueError("unexpected fresh candidate identity")
    if manifest.get("split_policy", {}).get("reserved_payload_loaded") is not False:
        raise ValueError("reserved payload must remain unopened")
    if not str(manifest.get("grouping", {}).get("scene_or_layout_identity_status", "")).startswith(
        "UNKNOWN"
    ):
        raise ValueError("scene/layout limitation must remain explicit")

    rows = manifest.get("episode_assignments")
    if not isinstance(rows, list):
        raise ValueError("episode assignments are required")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode = row.get("episode_index")
        if not isinstance(episode, int) or isinstance(episode, bool) or episode in by_id:
            raise ValueError("episode assignments contain an invalid or duplicate ID")
        by_id[episode] = row

    selection = manifest.get("pilot_selection", {})
    train_ids = selection.get("train_episode_indices", [])
    validation_ids = selection.get("validation_episode_indices", [])
    reserve_ids = manifest.get("reserved_development_episode_indices", [])
    sets = [set(train_ids), set(validation_ids), set(reserve_ids)]
    original_lists = (train_ids, validation_ids, reserve_ids)
    if any(
        len(ids) != len(values)
        for ids, values in zip(sets, original_lists, strict=True)
    ):
        raise ValueError("selected ID list contains duplicates")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("fresh partitions overlap")
    prior = manifest.get("prior_pilot_exposure", {})
    prior_ids = set(prior.get("train_ids", [])) | set(prior.get("validation_ids", []))
    if any(ids & prior_ids for ids in sets):
        raise ValueError("fresh partitions overlap prior pilot exposure")

    selected_by_split = {
        "train": train_ids,
        "validation": validation_ids,
        "reserved_holdout": reserve_ids,
    }
    for split, ids in selected_by_split.items():
        selected_rows = []
        for episode in ids:
            row = by_id.get(episode)
            if row is None or row.get("split") != split or row.get("source_split") != "train":
                raise ValueError(f"{split} ID {episode} is not a fresh source-train episode")
            selected_rows.append(row)
        if manifest.get("split_hashes", {}).get(
            "reserved_development" if split == "reserved_holdout" else split
        ) != _split_hash(selected_rows):
            raise ValueError(f"{split} split hash mismatch")

    counts = manifest.get("split_counts", {})
    actual_counts = {
        split: sum(row.get("split") == split for row in rows)
        for split in ("train", "validation", "reserved_holdout", "excluded")
    }
    if counts != actual_counts:
        raise ValueError("split counts do not match episode assignments")
    reasons = manifest.get("exclusion_reason_counts", {})
    actual_reasons = {
        reason: sum(row.get("exclusion_reason") == reason for row in rows)
        for reason in (
            "prior_pilot_exposed",
            "prior_validation_exposed",
            "prior_reserved_protected",
            "fresh_source_train_not_selected",
        )
    }
    if reasons != actual_reasons or sum(actual_reasons.values()) != counts["excluded"]:
        raise ValueError("exclusion reason counts do not match episode assignments")
    if len(train_ids) != selection.get("train_group_count"):
        raise ValueError("train group count mismatch")
    if len(validation_ids) != selection.get("validation_group_count"):
        raise ValueError("validation group count mismatch")
    if selection.get("train_episode_id_order_sha256") != _id_order_hash(
        [by_id[item] for item in train_ids]
    ):
        raise ValueError("train ID order hash mismatch")
    if selection.get("validation_episode_id_order_sha256") != _id_order_hash(
        [by_id[item] for item in validation_ids]
    ):
        raise ValueError("validation ID order hash mismatch")
    expected_download_bytes = sum(
        by_id[item]["video_bytes"] + by_id[item]["parquet_bytes"]
        for item in (*train_ids, *validation_ids)
    )
    if selection.get("selected_media_payload_bytes") != expected_download_bytes:
        raise ValueError("train/validation media byte estimate mismatch")


def build_fresh_manifest(
    source_manifest: dict[str, Any],
    *,
    train_groups: int = 4000,
    validation_groups: int = 1000,
    reserved_groups: int = 1000,
    seed: int = 17,
    protocol_id: str = "physicalai-stack-platforms-fresh-action-v1",
) -> dict[str, Any]:
    """Select fresh groups only from the original train split.

    The source manifest's pilot train and validation IDs are excluded. The
    original validation and reserved partitions are never eligible. The new
    reserved IDs are recorded but are not included in the downloader's
    train/validation selection.
    """
    counts = (train_groups, validation_groups, reserved_groups)
    if any(isinstance(value, bool) or value < 1 for value in counts):
        raise ValueError("all requested group counts must be positive")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    source = source_manifest["source"]
    if (
        source.get("dataset_id")
        != "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm"
        or source.get("revision") != "73c00c18e7ca108e91846294b20a1d7cd5de9340"
        or source.get("subset") != "panda-stack-platforms"
    ):
        raise ValueError("unexpected PhysicalAI source identity")
    if source_manifest.get("split_policy", {}).get("reserved_payload_loaded") is not False:
        raise ValueError("source manifest does not protect its reserved payload")

    assignments = source_manifest.get("episode_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("source manifest must contain episode assignments")
    rows_by_id: dict[int, dict[str, Any]] = {}
    for source_row in assignments:
        episode = source_row.get("episode_index")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValueError("invalid episode identity")
        if episode in rows_by_id:
            raise ValueError(f"duplicate source episode {episode}")
        fields = (
            "video_path",
            "video_bytes",
            "video_sha256",
            "parquet_path",
            "parquet_bytes",
            "parquet_sha256",
            "split",
        )
        for field in fields:
            if field not in source_row:
                raise ValueError(f"source episode {episode} lacks {field}")
        rows_by_id[episode] = dict(source_row)

    original_pilot = source_manifest.get("pilot_selection", {})
    prior_train = original_pilot.get("train_episode_indices", [])
    prior_validation = original_pilot.get("validation_episode_indices", [])
    if not isinstance(prior_train, list) or not isinstance(prior_validation, list):
        raise ValueError("source pilot IDs must be lists")
    prior_exposed = {int(item) for item in (*prior_train, *prior_validation)}
    if len(prior_exposed) != len(prior_train) + len(prior_validation):
        raise ValueError("source pilot train/validation IDs overlap or repeat")
    if not prior_exposed <= rows_by_id.keys():
        raise ValueError("source pilot references an unknown episode")
    if any(rows_by_id[item]["split"] != "train" for item in prior_train):
        raise ValueError("source pilot train IDs are not in source train")
    if any(rows_by_id[item]["split"] != "validation" for item in prior_validation):
        raise ValueError("source pilot validation IDs are not in source validation")

    eligible = [
        row
        for row in rows_by_id.values()
        if row["split"] == "train" and row["episode_index"] not in prior_exposed
    ]
    requested = sum(counts)
    if len(eligible) < requested:
        raise ValueError(
            f"only {len(eligible)} fresh source-train groups; {requested} requested"
        )
    ranked = sorted(eligible, key=lambda row: _rank_key(row, seed=seed, protocol_id=protocol_id))
    selected = {
        "train": ranked[:train_groups],
        "validation": ranked[train_groups : train_groups + validation_groups],
        "reserved_development": ranked[
            train_groups + validation_groups : train_groups + validation_groups + reserved_groups
        ],
    }
    selected_ids = {
        name: [row["episode_index"] for row in rows]
        for name, rows in selected.items()
    }
    all_selected = [item for ids in selected_ids.values() for item in ids]
    if len(all_selected) != len(set(all_selected)):
        raise AssertionError("fresh split selection is not disjoint")
    if set(all_selected) & prior_exposed:
        raise AssertionError("fresh split overlaps the prior pilot")

    candidate_split = {episode: "excluded" for episode in rows_by_id}
    for episode in selected_ids["train"]:
        candidate_split[episode] = "train"
    for episode in selected_ids["validation"]:
        candidate_split[episode] = "validation"
    for episode in selected_ids["reserved_development"]:
        candidate_split[episode] = "reserved_holdout"

    output_rows = []
    for episode, source_row in sorted(rows_by_id.items()):
        row = dict(source_row)
        row["source_split"] = source_row["split"]
        row["split"] = candidate_split[episode]
        if episode in prior_exposed:
            row["exclusion_reason"] = "prior_pilot_exposed"
        elif source_row["split"] == "validation":
            row["exclusion_reason"] = "prior_validation_exposed"
        elif source_row["split"] == "reserved_holdout":
            row["exclusion_reason"] = "prior_reserved_protected"
        elif candidate_split[episode] == "excluded":
            row["exclusion_reason"] = "fresh_source_train_not_selected"
        else:
            row["exclusion_reason"] = None
        output_rows.append(row)

    source_split_counts = source_manifest.get("split_counts", {})
    asset_hashes = source_manifest.get("asset_inventory_hashes", {})
    train_rows = selected["train"]
    validation_rows = selected["validation"]
    exclusion_reason_counts = {
        reason: sum(row["exclusion_reason"] == reason for row in output_rows)
        for reason in (
            "prior_pilot_exposed",
            "prior_validation_exposed",
            "prior_reserved_protected",
            "fresh_source_train_not_selected",
        )
    }
    manifest = {
        "schema_version": 1,
        "candidate_id": "nvidia-physicalai-singlearm-stack-platforms-fresh-action-v1",
        "status": "REVIEW_NOT_ADMITTED",
        "source": dict(source),
        "source_manifest_sha256": source_manifest.get("manifest_sha256"),
        "source_manifest_canonical_json_sha256": _sha256(
            json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()
        ),
        "source_split_counts": source_split_counts,
        "source_asset_inventory_hashes": asset_hashes,
        "grouping": dict(source_manifest["grouping"]),
        "split_policy": {
            "seed": seed,
            "rank": "sha256(protocol_id|seed|episode_index)",
            "protocol_id": protocol_id,
            "allocation": (
                "fresh source-train episodes only; prior pilot IDs excluded; "
                "original validation and reserved partitions excluded; "
                "fresh reserved_holdout cohort is not downloaded"
            ),
            "fresh_train_groups": train_groups,
            "fresh_validation_groups": validation_groups,
            "reserved_development_groups": reserved_groups,
            "source_pilot_train_episode_ids_exposed_before": len(prior_train),
            "source_pilot_validation_episode_ids_exposed_before": len(prior_validation),
            "source_validation_partition_groups_excluded": source_split_counts.get(
                "validation", 0
            ),
            "source_reserved_partition_groups_protected": source_split_counts.get(
                "reserved_holdout", 0
            ),
            "reserved_payload_loaded": False,
        },
        "split_counts": {
            "train": train_groups,
            "validation": validation_groups,
            "reserved_holdout": reserved_groups,
            "excluded": len(output_rows) - requested,
        },
        "exclusion_reason_counts": exclusion_reason_counts,
        "split_hashes": {
            name: _split_hash(rows) for name, rows in selected.items()
        },
        "episode_id_order_hashes": {
            name: _id_order_hash(rows) for name, rows in selected.items()
        },
        "prior_pilot_exposure": {
            "train_ids": sorted(int(item) for item in prior_train),
            "validation_ids": sorted(int(item) for item in prior_validation),
            "train_id_order_sha256": original_pilot.get("train_episode_id_order_sha256"),
            "validation_id_order_sha256": original_pilot.get("validation_episode_id_order_sha256"),
        },
        "selected_media_payload_bytes": {
            name: sum(row["video_bytes"] + row["parquet_bytes"] for row in rows)
            for name, rows in selected.items()
        },
        "pilot_selection": {
            "train_episode_indices": selected_ids["train"],
            "validation_episode_indices": selected_ids["validation"],
            "train_group_count": train_groups,
            "validation_group_count": validation_groups,
            "train_episode_id_order_sha256": _id_order_hash(train_rows),
            "validation_episode_id_order_sha256": _id_order_hash(validation_rows),
            "selected_media_payload_bytes": sum(
                row["video_bytes"] + row["parquet_bytes"]
                for row in train_rows + validation_rows
            ),
        },
        "reserved_development_episode_indices": selected_ids["reserved_development"],
        "episode_assignments": output_rows,
    }
    validate_fresh_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-groups", type=int, default=4000)
    parser.add_argument("--validation-groups", type=int, default=1000)
    parser.add_argument("--reserved-groups", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    source_bytes = args.source_manifest.read_bytes()
    source_sha256 = _sha256(source_bytes)
    if source_sha256 != PINNED_SOURCE_MANIFEST_SHA256:
        parser.error("source manifest hash differs from the pinned parent manifest")
    source_manifest = json.loads(source_bytes)
    source_manifest["manifest_sha256"] = source_sha256
    result = build_fresh_manifest(
        source_manifest,
        train_groups=args.train_groups,
        validation_groups=args.validation_groups,
        reserved_groups=args.reserved_groups,
        seed=args.seed,
    )
    validate_fresh_manifest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_groups": result["split_counts"]["train"],
                "validation_groups": result["split_counts"]["validation"],
                "reserved_groups": result["split_counts"]["reserved_holdout"],
                "train_validation_bytes": result["pilot_selection"][
                    "selected_media_payload_bytes"
                ],
                "reserved_bytes_not_downloaded": result[
                    "selected_media_payload_bytes"
                ]["reserved_development"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
