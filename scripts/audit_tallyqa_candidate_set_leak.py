"""Audit whether the legacy TallyQA option set leaks its count label."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.prepare_tallyqa_vg_task import options_for
from tiny_omni_decision.dataset import sha256_file

EXPECTED_TASK_SHA256 = "59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45"


def summarize_train_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels_by_set: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for row in rows:
        values = tuple(sorted(int(option["value"]) for option in row["options"]))
        answer = int(row["answer"])
        if answer not in values:
            raise ValueError("answer is not present in its candidate set")
        labels_by_set[values][answer] += 1
    if not rows:
        raise ValueError("train rows are required")

    unambiguous_rows = sum(
        sum(counts.values()) for counts in labels_by_set.values() if len(counts) == 1
    )
    majority_correct = sum(max(counts.values()) for counts in labels_by_set.values())
    policy_labels_by_set: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for answer in range(16):
        policy_labels_by_set[tuple(sorted(options_for(answer)))].add(answer)
    uniquely_recoverable_labels = sorted(
        answer
        for values, answers in policy_labels_by_set.items()
        if len(answers) == 1
        for answer in answers
    )
    return {
        "train_rows": len(rows),
        "distinct_candidate_sets": len(labels_by_set),
        "train_rows_in_sets_with_one_observed_train_label": unambiguous_rows,
        "fraction_train_rows_in_sets_with_one_observed_train_label": unambiguous_rows / len(rows),
        "in_sample_train_majority_candidate_set_correct": majority_correct,
        "in_sample_train_majority_candidate_set_accuracy": majority_correct / len(rows),
        "policy_uniquely_reveals_possible_labels": uniquely_recoverable_labels,
        "validation_rows_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_sha = sha256_file(args.task_manifest)
    if actual_sha != EXPECTED_TASK_SHA256:
        raise ValueError("legacy task manifest SHA-256 differs from the pinned source")
    task = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    if task.get("task", {}).get("name") != "tallyqa_vg_amt_four_option_counting":
        raise ValueError("unexpected legacy task ID")
    result = {
        "schema_version": 1,
        "status": "candidate_set_shortcut_confirmed_train_only",
        "task_manifest_sha256": actual_sha,
        "task_id": task["task"]["name"],
        "train_only_diagnostic": summarize_train_rows(task["train"]),
        "official_test_opened": False,
        "sealed_audit_opened": False,
        "interpretation": (
            "The majority candidate-set score is in-sample on the training rows. "
            "It demonstrates a shortcut in the option construction, not held-out quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
