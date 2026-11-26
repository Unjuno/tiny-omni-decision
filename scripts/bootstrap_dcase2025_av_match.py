"""Paired parent/room bootstrap of saved four-choice AV-match predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_sums(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    prediction = probabilities.argmax(1)
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1.0))
    brier = np.square(probabilities - np.eye(4)[targets]).sum(1)
    return np.column_stack((prediction == targets, nll, brier)).astype(np.float64)


def summarize(run_dir: Path, *, replicates: int = 10_000, seed: int = 17):
    output_path = run_dir / "paired-bootstrap.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    config = json.loads((run_dir / "effective-config.json").read_text(encoding="utf-8"))
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions_path = run_dir / "oof-predictions.jsonl"
    if (
        result.get("status") != "PASS"
        or result.get("selection_used_outer_test") is not False
        or result.get("development_test_accessed") is not False
        or result.get("sealed_audit_accessed") is not False
        or config.get("outer_protocol")
        != "leave-one-room-out across 9 rooms; every query tested once"
        or sha256_file(predictions_path) != result.get("predictions_sha256")
    ):
        raise ValueError("saved AV-match run integrity gate failed")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != result.get("queries") or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("saved prediction rows are incomplete or duplicated")
    if replicates < 1 or seed < 0:
        raise ValueError("replicates must be positive and seed non-negative")
    targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    if not np.isin(targets, np.arange(4)).all():
        raise ValueError("invalid four-choice target")
    arms = ("joint", "audio_missing", "video_missing", "video_shuffled", "uniform")
    probabilities = {
        arm: np.asarray([row["probabilities"][arm] for row in rows], dtype=np.float64)
        for arm in arms
    }
    if any(
        value.shape != (len(rows), 4)
        or not np.isfinite(value).all()
        or not np.allclose(value.sum(1), 1.0, atol=1e-7)
        for value in probabilities.values()
    ):
        raise ValueError("invalid four-choice probability array")
    comparisons = {}
    rng = np.random.default_rng(seed)
    for group_name, group_ids in (
        ("parent_recording", [str(row["query_parent"]) for row in rows]),
        ("room", [str(row["room_id"]) for row in rows]),
    ):
        groups = sorted(set(group_ids))
        group_positions = np.asarray(group_ids, dtype=object)
        group_indices = [np.flatnonzero(group_positions == group) for group in groups]
        draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
        denominators = np.asarray([len(indices) for indices in group_indices])[draws].sum(1)
        arm_sums = {
            arm: np.stack(
                [_row_sums(probabilities[arm][idx], targets[idx]).sum(0) for idx in group_indices]
            )
            for arm in arms
        }
        arm_comparisons = {}
        for baseline in ("uniform", "video_shuffled", "audio_missing", "video_missing"):
            delta = arm_sums["joint"] - arm_sums[baseline]
            point = delta.sum(0) / len(rows)
            sampled = delta[draws].sum(1) / denominators[:, None]
            interval = np.quantile(sampled, [0.025, 0.975], axis=0)
            arm_comparisons[f"joint_minus_{baseline}"] = {
                "point": {
                    "accuracy": float(point[0]),
                    "nll": float(point[1]),
                    "brier": float(point[2]),
                },
                "percentile_95_ci": {
                    name: [float(value) for value in interval[:, index]]
                    for index, name in enumerate(("accuracy", "nll", "brier"))
                },
            }
        comparisons[group_name] = {
            "independent_groups": len(groups),
            "replicates": replicates,
            "seed": seed,
            "contrasts": arm_comparisons,
        }
    report = {
        "status": "PASS",
        "scope": (
            "posthoc paired bootstrap of saved OOF predictions only; "
            "no fitting or devtest access"
        ),
        "run_result_sha256": sha256_file(result_path),
        "oof_predictions_sha256": result["predictions_sha256"],
        "queries": len(rows),
        "replicates": replicates,
        "seed": seed,
        "paired_group_differences": comparisons,
        "interpretation_limit": (
            "Only 78 query parents and 9 rooms; development diagnostic, "
            "not an independent audit."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = summarize(args.run_dir, replicates=args.replicates, seed=args.seed)
    print(json.dumps({"status": report["status"], "queries": report["queries"]}))


if __name__ == "__main__":
    main()
