"""Audit a pinned sample of new DROID shards without reading labels or media."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.audit_droid100_groups import (  # noqa: E402
    episode_example_context,
    episode_example_feature_counts,
    iter_tfrecord,
)
from tiny_omni_decision.droid_site_holdout import (  # noqa: E402
    droid_site_from_path,
    select_unseen_droid_shards,
    stable_droid_id,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), raw


def input_receipts(paths: list[Path]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        value, _ = read_json(path)
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict) or "index" not in row:
                raise ValueError(f"receipt row lacks a shard index: {path}")
            index = int(row["index"])
            if index in result:
                raise ValueError(f"duplicate receipt for shard {index}")
            result[index] = row
    return result


def parse_input(value: str) -> tuple[int, Path]:
    try:
        index_text, path_text = value.split("=", 1)
        index = int(index_text)
    except (ValueError, TypeError) as error:
        raise ValueError("input must be INDEX=PATH") from error
    if index < 0 or not path_text:
        raise ValueError("input must include a non-negative index and path")
    return index, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="append", required=True, metavar="INDEX=PATH")
    parser.add_argument("--receipts", action="append", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit result: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("audit output must remain outside the repository")

    selection, selection_raw = read_json(args.selection)
    expected_sample = select_unseen_droid_shards()
    sample_indices = selection.get("indices")
    if sample_indices != expected_sample:
        raise ValueError("selection manifest does not match the pinned hash sample")

    previous_rows = [
        json.loads(line)
        for line in args.previous_cohort.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previous_ids = {row["id"] for row in previous_rows}
    previous_groups = {row["group_id"] for row in previous_rows}
    previous_sites = {row["site"] for row in previous_rows}
    if (
        len(previous_ids) != len(previous_rows)
        or len(previous_groups) != len(previous_rows)
        or len(previous_sites) < 1
    ):
        raise ValueError("previous cohort identities are duplicated or incomplete")

    info, _ = read_json(args.dataset_info)
    split = next(row for row in info["splits"] if row["name"] == "train")
    shard_lengths = [int(length) for length in split["shardLengths"]]
    receipts = input_receipts(args.receipts)
    inputs = [parse_input(item) for item in args.inputs]
    indices = [index for index, _ in inputs]
    if len(set(indices)) != len(indices) or len(indices) < len(expected_sample):
        raise ValueError("input shard indices must be unique and include the selected sample")
    if not set(expected_sample).issubset(indices):
        raise ValueError("inputs do not include every preselected sample shard")

    started = time.perf_counter()
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_payloads: set[str] = set()
    prior_id_overlap: set[str] = set()
    prior_group_overlap: set[str] = set()
    duplicate_ids: set[str] = set()
    duplicate_groups: set[str] = set()
    duplicate_payloads: set[str] = set()
    site_counts: Counter[str] = Counter()
    eligible_site_counts: Counter[str] = Counter()
    new_sample_sites: set[str] = set()
    shard_rows = []

    for index, path in inputs:
        if index >= len(shard_lengths):
            raise ValueError(f"shard index outside pinned dataset metadata: {index}")
        receipt = receipts.get(index)
        if receipt is None:
            raise ValueError(f"missing integrity receipt for shard {index}")
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
            raise ValueError(f"input shard missing or wrong size: {path}")
        digest = sha256_file(path)
        if digest != str(receipt["sha256"]).lower():
            raise ValueError(f"input shard SHA-256 mismatch: {path}")

        record_count = 0
        eligible_count = 0
        shard_sites: set[str] = set()
        for raw in iter_tfrecord(path):
            record_count += 1
            context = episode_example_context(raw)
            source_path = context["episode_metadata/file_path"]
            group_path = context["episode_metadata/recording_folderpath"]
            source_id = stable_droid_id(source_path)
            group_id = stable_droid_id(group_path)
            site = droid_site_from_path(source_path)
            feature_counts = episode_example_feature_counts(raw)
            action_count = int(
                feature_counts["steps/action_dict/cartesian_velocity"]["count"]
            )
            if action_count % 6:
                raise ValueError("DROID action sequence length is not divisible by six")
            source_steps = action_count // 6
            eligible = source_steps >= 32 and int(source_steps * 0.25) >= 8

            if source_id in seen_ids:
                duplicate_ids.add(source_id)
            if group_id in seen_groups:
                duplicate_groups.add(group_id)
            payload_id = hashlib.sha256(raw).hexdigest()
            if payload_id in seen_payloads:
                duplicate_payloads.add(payload_id)
            if source_id in previous_ids:
                prior_id_overlap.add(source_id)
            if group_id in previous_groups:
                prior_group_overlap.add(group_id)

            seen_ids.add(source_id)
            seen_groups.add(group_id)
            seen_payloads.add(payload_id)
            site_counts[site] += 1
            eligible_site_counts[site] += int(eligible)
            shard_sites.add(site)
            eligible_count += int(eligible)
            if index in expected_sample and site not in previous_sites:
                new_sample_sites.add(site)

        if record_count != shard_lengths[index]:
            raise ValueError(
                f"shard {index} record count {record_count} != pinned metadata "
                f"{shard_lengths[index]}"
            )
        shard_rows.append(
            {
                "index": index,
                "records": record_count,
                "eligible_records": eligible_count,
                "site_count": len(shard_sites),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    result = {
        "status": "PASS_SOURCE_METADATA_AND_IDENTITY",
        "selection_indices": expected_sample,
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "previous_cohort_sha256": sha256_file(args.previous_cohort),
        "dataset_info_sha256": sha256_file(args.dataset_info),
        "record_count_in_selected_shards": sum(
            row["records"] for row in shard_rows if row["index"] in expected_sample
        ),
        "eligible_count_in_selected_shards": sum(
            row["eligible_records"]
            for row in shard_rows
            if row["index"] in expected_sample
        ),
        "unique_source_ids": len(seen_ids),
        "unique_recording_groups": len(seen_groups),
        "prior_cohort_id_overlap": len(prior_id_overlap),
        "prior_cohort_group_overlap": len(prior_group_overlap),
        "duplicate_ids_across_inputs": len(duplicate_ids),
        "duplicate_groups_across_inputs": len(duplicate_groups),
        "exact_record_duplicates_across_inputs": len(duplicate_payloads),
        "existing_site_count": len(previous_sites),
        "new_sites_in_selected_sample": sorted(new_sample_sites),
        "site_counts": dict(sorted(site_counts.items())),
        "eligible_site_counts": dict(sorted(eligible_site_counts.items())),
        "shards": shard_rows,
        "labels_read": False,
        "media_decoded": False,
        "sealed_audit_loaded": False,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
