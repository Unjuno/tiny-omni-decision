"""Count adjacent command changes in previously frozen train recordings."""

import argparse
import json
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import (
    OPTIONS,
    RECEIPT_SHA,
    TRAIN_SHA,
    canonical_id,
    digest,
    direction,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
DATA = ROOT / "droid-fresh-next-action-seed17-attempt-01"
TRAIN = ROOT / "droid-fresh-next-action-prep-attempt-02/train.jsonl"
PARSER = ROOT / "droid-train-temporal-baselines-20261027-attempt-01/pinned-parser.py"
PARSER_SHA = "1d5d7da7bba74c5b4ebc13636efbbd9cce2c510a90509ac744fefc268fb5b659"
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_TRAIN_ALL_TRANSITIONS_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def transition_counts(actions, minimum_target=8):
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError("actions must be finite [steps,6]")
    if minimum_target < 1:
        raise ValueError("target must follow an observed command")
    labels = [direction(row[:3]) for row in array]
    matrix = np.zeros((7, 7), dtype=np.int64)
    for index in range(minimum_target, len(labels)):
        matrix[labels[index - 1], labels[index]] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output required")
    receipt_path = DATA / "download-receipt.json"
    pinned = {receipt_path: RECEIPT_SHA, TRAIN: TRAIN_SHA, PARSER: PARSER_SHA}
    if any(digest(path) != value for path, value in pinned.items()):
        raise ValueError("pinned input hash mismatch")
    train = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines()]
    if (len(train) != 275 or any(row["split"] != "train" for row in train)
            or len({row["id"] for row in train}) != 275
            or len({row["group_id"] for row in train}) != 275):
        raise ValueError("unexpected fixed train cohort")
    by_id = {row["id"]: row for row in train}
    receipt = json.loads(receipt_path.read_bytes())
    api = runpy.run_path(str(PARSER), run_name="pinned_droid_parser")
    seen, skipped, shard_count = set(), 0, 0
    matrix = np.zeros((7, 7), dtype=np.int64)
    episodes_by_target = np.zeros(7, dtype=np.int64)
    change_distribution = Counter()
    start = time.perf_counter()
    for item in receipt["objects"]:
        if ".tfrecord-" not in item["name"]:
            continue
        path = DATA / item["name"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("pinned source shard mismatch")
        shard_count += 1
        for raw in api["iter_tfrecord"](path):
            context = api["episode_example_context"](raw)
            identifier = canonical_id(context["episode_metadata/file_path"])
            if identifier not in by_id:
                skipped += 1
                continue
            if identifier in seen:
                raise ValueError("duplicate train recording")
            row = by_id[identifier]
            if canonical_id(context["episode_metadata/recording_folderpath"]) != row["group_id"]:
                raise ValueError("recording identity mismatch")
            actions = np.asarray(api["episode_example_float_values"](
                raw, "steps/action_dict/cartesian_velocity"), dtype=np.float64).reshape(-1, 6)
            if len(actions) != row["source_step_count"]:
                raise ValueError("source step count mismatch")
            if OPTIONS[direction(actions[row["target_index"], :3])] != row["target_option_id"]:
                raise ValueError("fixed target mismatch")
            counts = transition_counts(actions)
            matrix += counts
            changes = counts.copy()
            np.fill_diagonal(changes, 0)
            change_distribution[int(changes.sum())] += 1
            episodes_by_target += (changes.sum(axis=0) > 0).astype(np.int64)
            seen.add(identifier)
        print(f"processed {shard_count} shards; {len(seen)} train recordings", flush=True)
    if seen != set(by_id) or any(digest(path) != value for path, value in pinned.items()):
        raise ValueError("missing recordings or changed source metadata")
    total, same = int(matrix.sum()), int(np.trace(matrix))
    result = {
        "status": "TRAIN_RECORDING_FEASIBILITY_ONLY", "recordings": len(seen),
        "shards": shard_count, "opaque_nonselected_records_skipped": skipped,
        "minimum_target_step": 8, "command_threshold": 1e-4,
        "transition_pairs": total, "same_pairs": same, "changed_pairs": total - same,
        "previous_command_copy_accuracy": same / total if total else None,
        "transition_matrix_rows_previous_columns_next": matrix.tolist(),
        "next_class_counts": dict(zip(OPTIONS, matrix.sum(axis=0).tolist(), strict=True)),
        "recordings_with_changed_next_class": dict(
            zip(OPTIONS, episodes_by_target.tolist(), strict=True)
        ),
        "recordings_by_change_count": dict(sorted(change_distribution.items())),
        "recordings_with_any_change": len(seen) - change_distribution[0],
        "input_hashes": {str(path): value for path, value in pinned.items()},
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "wall_seconds": time.perf_counter() - start,
        "limitations": (
            "correlated train timesteps; opaque nonselected records were read but not decoded"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
