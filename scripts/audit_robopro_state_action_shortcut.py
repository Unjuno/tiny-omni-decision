"""Measure a train-only current-state/action retrieval shortcut in RoboPro."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXPECTED_SAMPLE_SHA256 = "eeab697bc4189ba2b4442ed53def960f248cf146b2cc947c0633905a41340686"
EXPECTED_SPLIT_SHA256 = "b90196fd2963a5eb90f6e41339f71129c6306fbfd7b45fd80326a5ebdd4c60c1"
EXPECTED_RECEIPT_SHA256 = "3c5da58198c80478526cde6513c29cd58954416cf19244a715bfeeec2c275ce3"
EXPECTED_EPISODES_SHA256 = "88492784987d483618b5144f37bac8c3036bdaaa11d2df730f7b26a804b68658"
EXPECTED_INFO_SHA256 = "91c7d161a416685b08f9db5b195a6ba02f7047d966fa188e0262f854af386cfe"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midranks(values: Sequence[float]) -> list[float]:
    """Return ascending one-based average ranks, preserving exact ties."""
    if not values:
        raise ValueError("cannot rank an empty sequence")
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average_rank
        start = end
    return result


def score_prompt_group(
    episodes: Sequence[dict[str, Any]],
    *,
    dimensions: Sequence[int],
    progress_points: Sequence[float],
) -> list[dict[str, Any]]:
    """Rank each episode's logged command against same-prompt episode commands."""
    if len(episodes) != 4:
        raise ValueError("the frozen probe requires exactly four episodes per prompt")
    results = []
    for progress in progress_points:
        for query_index, episode in enumerate(episodes):
            query_row = round((len(episode["states"]) - 1) * progress)
            state = episode["states"][query_row]
            candidate_rows = [
                round((len(candidate_episode["actions"]) - 1) * progress)
                for candidate_episode in episodes
            ]
            candidates = [
                candidate_episode["actions"][candidate_row]
                for candidate_episode, candidate_row in zip(episodes, candidate_rows, strict=True)
            ]
            if any(
                len(state) <= dim or len(candidate) <= dim
                for candidate in candidates
                for dim in dimensions
            ):
                raise ValueError("state/action vector is shorter than a selected dimension")

            rank_sums = [0.0] * len(candidates)
            for dimension in dimensions:
                errors = [
                    abs(float(candidate[dimension]) - float(state[dimension]))
                    for candidate in candidates
                ]
                ranks = midranks(errors)
                rank_sums = [total + rank for total, rank in zip(rank_sums, ranks, strict=True)]

            best_score = min(rank_sums)
            winners = [index for index, score in enumerate(rank_sums) if score == best_score]
            results.append(
                {
                    "progress": progress,
                    "query_episode_id": episode["episode_id"],
                    "query_row_index": query_row,
                    "target_option_index": query_index,
                    "rank_sums": rank_sums,
                    "winner_indices": winners,
                    "top_tie": len(winners) > 1,
                    "fractional_credit": 1.0 / len(winners) if query_index in winners else 0.0,
                }
            )
    return results


def validate_frozen_split(sample: dict[str, Any], split_manifest: dict[str, Any]) -> None:
    """Require the frozen sample's train/development identities to match the split."""
    partitions: dict[str, tuple[set[int], set[str]]] = {}
    for name in ("train", "development"):
        records = sample.get(name)
        partition = split_manifest.get(name)
        if not isinstance(records, list) or len(records) != 16 or not isinstance(partition, dict):
            raise ValueError(f"unexpected frozen {name} sample structure")
        partition_ids = set(map(int, partition["episode_ids"]))
        partition_prompts = {
            sha256_bytes(prompt.encode("utf-8")) for prompt in partition["prompt_groups"]
        }
        group_ids: set[int] = set()
        group_prompts: set[str] = set()
        for group in records:
            prompt = str(group["prompt"])
            prompt_hash = sha256_bytes(prompt.encode("utf-8"))
            if prompt_hash != group["prompt_sha256"]:
                raise ValueError(f"{name} sample prompt hash mismatch")
            if prompt_hash in group_prompts or prompt_hash not in partition_prompts:
                raise ValueError(f"{name} sample prompts do not match unique split groups")
            group_prompts.add(prompt_hash)
            ids = {int(row["episode_id"]) for row in group["episodes"]}
            if len(ids) != 4 or not ids <= partition_ids or group_ids.intersection(ids):
                raise ValueError(f"{name} sample episodes do not match unique split members")
            group_ids.update(ids)
        if len(group_ids) != 64:
            raise ValueError(f"unexpected frozen {name} episode count")
        partitions[name] = (group_ids, group_prompts)

    train_ids, train_prompts = partitions["train"]
    development_ids, development_prompts = partitions["development"]
    if train_ids & development_ids or train_prompts & development_prompts:
        raise ValueError("frozen train/development sample identities overlap")


def run_audit(
    *,
    sample_path: Path,
    split_manifest_path: Path,
    data_root: Path,
    metadata_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify the pinned TRAIN slice and evaluate a no-fit retrieval baseline."""
    import pyarrow.parquet as pq

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    started = time.perf_counter()
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    sample_sha = sha256_bytes(sample_bytes)
    if sample_sha != EXPECTED_SAMPLE_SHA256:
        raise ValueError("sample manifest does not match the pinned frozen sample")
    if sample.get("schema_version") != "robopro-action-probe-sample-v1":
        raise ValueError("unexpected sample manifest schema")
    if sample.get("revision") != "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0":
        raise ValueError("unexpected RoboPro source revision")
    if len(sample.get("train", [])) != 16 or sample.get("development") is None:
        raise ValueError("sample does not match the frozen 16-prompt train/development design")

    split_bytes = split_manifest_path.read_bytes()
    split_sha = sha256_bytes(split_bytes)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("prompt split does not match the pinned frozen split")
    split_manifest = json.loads(split_bytes)
    if split_sha != sample.get("split_manifest_sha256"):
        raise ValueError("sample manifest does not bind to the frozen prompt split")
    if split_manifest.get("revision") != sample["revision"]:
        raise ValueError("split/source revision mismatch")
    validate_frozen_split(sample, split_manifest)

    source_episodes_bytes = (metadata_dir / "episodes.jsonl").read_bytes()
    if (
        sha256_bytes(source_episodes_bytes) != EXPECTED_EPISODES_SHA256
        or sha256_bytes(source_episodes_bytes) != sample.get("source_metadata_sha256")
    ):
        raise ValueError("source episode metadata does not match the frozen sample")

    receipt_path = data_root / "download-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if sha256_bytes(receipt_bytes) != EXPECTED_RECEIPT_SHA256:
        raise ValueError("download receipt does not match the pinned frozen receipt")
    if receipt.get("sample_manifest_sha256") != sample_sha:
        raise ValueError("download receipt does not bind to the supplied sample manifest")
    if receipt.get("revision") != sample["revision"]:
        raise ValueError("receipt/source revision mismatch")
    root = data_root.resolve()
    train_entries = {}
    for row in receipt.get("files", []):
        if row.get("split") != "train":
            continue
        key = (int(row["episode_id"]), row["kind"])
        if key in train_entries:
            raise ValueError(f"duplicate train receipt entry: {key}")
        train_entries[key] = row

    info_bytes = (metadata_dir / "info.json").read_bytes()
    if sha256_bytes(info_bytes) != EXPECTED_INFO_SHA256:
        raise ValueError("source info metadata does not match the pinned revision")
    info = json.loads(info_bytes)
    state_names = info["features"]["observation.state"]["names"][0]
    action_names = info["features"]["action"]["names"][0]
    if state_names != action_names or len(state_names) != 14:
        raise ValueError("publisher state/action channel names are not aligned")
    dimensions = [index for index, name in enumerate(state_names) if not name.endswith("gripper")]
    if len(dimensions) != 12:
        raise ValueError("expected exactly twelve named angular channels")

    prompt_rows = []
    file_checks = []
    all_query_results = []
    progress_points = (0.25, 0.5, 0.75)
    for group in sample["train"]:
        episodes = []
        episode_ids = [int(row["episode_id"]) for row in group["episodes"]]
        if len(episode_ids) != 4 or len(set(episode_ids)) != 4:
            raise ValueError("each train prompt must contain four distinct episodes")
        for record in group["episodes"]:
            episode_id = int(record["episode_id"])
            receipt_row = train_entries.get((episode_id, "parquet"))
            if receipt_row is None:
                raise ValueError(f"missing train-only Parquet receipt: episode {episode_id}")
            relative = Path(receipt_row["relative_path"])
            parquet_path = (root / relative).resolve()
            if not parquet_path.is_relative_to(root):
                raise ValueError("receipt path escapes the data root")
            payload_hash = sha256_file(parquet_path)
            if payload_hash != receipt_row["sha256"] or parquet_path.stat().st_size != int(
                receipt_row["bytes"]
            ):
                raise ValueError(f"Parquet receipt integrity mismatch: episode {episode_id}")
            table = pq.read_table(
                parquet_path, columns=["observation.state", "action", "episode_index"]
            )
            columns = table.to_pydict()
            if any(int(value) != episode_id for value in columns["episode_index"]):
                raise ValueError(f"Parquet episode id mismatch: {episode_id}")
            states = columns["observation.state"]
            actions = columns["action"]
            if len(states) != int(record["length"]) or len(actions) != len(states):
                raise ValueError(f"Parquet row count mismatch: episode {episode_id}")
            if any(len(row) != 14 for row in (*states, *actions)):
                raise ValueError(f"state/action dimension mismatch: episode {episode_id}")
            if any(not math.isfinite(float(value)) for row in (*states, *actions) for value in row):
                raise ValueError(f"non-finite state/action value: episode {episode_id}")
            episodes.append({"episode_id": episode_id, "states": states, "actions": actions})
            file_checks.append(
                {
                    "episode_id": episode_id,
                    "bytes": int(receipt_row["bytes"]),
                    "sha256": payload_hash,
                }
            )

        group_results = score_prompt_group(
            episodes, dimensions=dimensions, progress_points=progress_points
        )
        all_query_results.extend(group_results)
        credited = sum(float(row["fractional_credit"]) for row in group_results)
        prompt_rows.append(
            {
                "prompt_sha256": group["prompt_sha256"],
                "queries": len(group_results),
                "fractional_correct": credited,
                "accuracy_fractional": credited / len(group_results),
                "top_ties": sum(bool(row["top_tie"]) for row in group_results),
            }
        )

    total_queries = len(all_query_results)
    total_credit = sum(float(row["fractional_credit"]) for row in all_query_results)
    report = {
        "schema_version": "robopro-state-action-shortcut-audit-v1",
        "status": "train_only_no_fit",
        "dataset": sample["dataset"],
        "revision": sample["revision"],
        "sample_manifest_sha256": sample_sha,
        "split_manifest_sha256": split_sha,
        "source_episodes_metadata_sha256": sha256_bytes(source_episodes_bytes),
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "info_sha256": sha256_bytes(info_bytes),
        "split_used": "train",
        "development_rows_opened": False,
        "model_fit_or_inference": False,
        "algorithm": {
            "name": "same-progress current-state-to-candidate-command rank retrieval",
            "progress_points": list(progress_points),
            "ranking": (
                "sum per-channel ascending midranks of absolute state-command "
                "error; ties receive fractional credit"
            ),
            "dimensions": [{"index": index, "name": state_names[index]} for index in dimensions],
            "excluded_dimensions": [
                {"index": index, "name": state_names[index]}
                for index in range(14)
                if index not in dimensions
            ],
            "assumption": (
                "same-name state and target-command channels share monotone "
                "numeric encoding; no semantic policy validity is asserted"
            ),
        },
        "counts": {
            "prompt_groups": len(prompt_rows),
            "episodes": len(file_checks),
            "queries": total_queries,
            "top_ties": sum(bool(row["top_tie"]) for row in all_query_results),
        },
        "fractional_accuracy": total_credit / total_queries,
        "fractional_correct": total_credit,
        "by_prompt": prompt_rows,
        "queries_detail": all_query_results,
        "verified_train_parquets": file_checks,
        "runtime": {"seconds": time.perf_counter() - started, "device": "CPU"},
        "limitations": [
            "train-only shortcut diagnostic; not a generalization estimate",
            "same-encoding assumption is supported only by matching published "
            "channel names, not a unit declaration",
            "recorded demonstrator command is not proven uniquely correct or optimal",
            "exact-prompt-held-out development results remain exposed "
            "development evidence and were not opened by this audit",
            "no physical-scene generalization is established",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        sample_path=args.sample_manifest,
        split_manifest_path=args.split_manifest,
        data_root=args.data_root,
        metadata_dir=args.metadata_dir,
        output_path=args.output,
    )
    print(
        f"{report['counts']['queries']} train queries; "
        f"fractional accuracy={report['fractional_accuracy']:.6f}; "
        f"top ties={report['counts']['top_ties']}"
    )


if __name__ == "__main__":
    main()
