"""Audit repeated/conflicting model-visible HABIT text inputs on train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("question and options must be strings")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("question and options must be nonempty")
    return normalized


def _model_visible_text_key(record: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    question = _normalize_text(record["question"])
    options = tuple(_normalize_text(option) for option in record["options"])
    target_index = record["target_index"]
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("options must be unique after normalization")
    if not isinstance(target_index, int) or not 0 <= target_index < len(options):
        raise ValueError("target_index is outside the supplied options")
    return question, tuple(sorted(options)), options[target_index]


def audit_train_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") != "train":
            continue
        question, options, target = _model_visible_text_key(record)
        buckets[(question, options)].append({
            "id": str(record["id"]),
            "target": target,
            "sid": str(record["sid"]),
            "episode_index": int(record["episode_index"]),
            "video_sha256": str(record["video_sha256"]),
        })

    train_count = sum(len(rows) for rows in buckets.values())
    repeated = {key: rows for key, rows in buckets.items() if len(rows) > 1}
    conflicts = {
        key: rows for key, rows in repeated.items()
        if len({row["target"] for row in rows}) > 1
    }

    def summarize(key: tuple[str, tuple[str, ...]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        targets = Counter(row["target"] for row in rows)
        key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        return {
            "input_key_sha256": hashlib.sha256(key_json.encode("utf-8")).hexdigest(),
            "records": len(rows),
            "target_counts": dict(sorted(targets.items())),
            "majority_target_count": max(targets.values()),
            "distinct_task_groups": len({row["sid"] for row in rows}),
            "distinct_episodes": len({row["episode_index"] for row in rows}),
            "distinct_videos": len({row["video_sha256"] for row in rows}),
            "sample_ids": sorted(row["id"] for row in rows),
        }

    repeated_records = sum(len(rows) for rows in repeated.values())
    conflicting_records = sum(len(rows) for rows in conflicts.values())
    repeated_majority_correct = sum(
        max(Counter(row["target"] for row in rows).values())
        for rows in repeated.values()
    )
    conflict_majority_correct = sum(
        max(Counter(row["target"] for row in rows).values())
        for rows in conflicts.values()
    )
    conflict_majority_total = sum(len(rows) for rows in conflicts.values())
    cross_episode_conflicts = {
        key: rows for key, rows in conflicts.items()
        if len({row["episode_index"] for row in rows}) > 1
    }
    cross_video_conflicts = {
        key: rows for key, rows in conflicts.items()
        if len({row["video_sha256"] for row in rows}) > 1
    }

    return {
        "status": "TRAIN_ONLY_TEXT_IDENTIFIABILITY_AUDIT_COMPLETE",
        "normalization": "Unicode NFKC, casefold, collapse whitespace; preserve punctuation",
        "input_key": (
            "normalized question + sorted normalized option texts; target mapped to "
            "semantic option text"
        ),
        "train_examples": train_count,
        "unique_model_visible_text_keys": len(buckets),
        "repeated_text_keys": len(repeated),
        "records_in_repeated_text_keys": repeated_records,
        "conflicting_text_keys": len(conflicts),
        "records_in_conflicting_text_keys": conflicting_records,
        "conflicting_target_assignments_beyond_majority": (
            conflict_majority_total - conflict_majority_correct
        ),
        "cross_episode_conflict_keys": len(cross_episode_conflicts),
        "records_in_cross_episode_conflicts": sum(
            len(rows) for rows in cross_episode_conflicts.values()
        ),
        "cross_video_conflict_keys": len(cross_video_conflicts),
        "records_in_cross_video_conflicts": sum(
            len(rows) for rows in cross_video_conflicts.values()
        ),
        "repeated_key_majority_oracle": {
            "correct": repeated_majority_correct,
            "count": repeated_records,
            "accuracy": repeated_majority_correct / repeated_records
            if repeated_records else None,
        },
        "conflict_key_majority_oracle": {
            "correct": conflict_majority_correct,
            "count": conflict_majority_total,
            "accuracy": conflict_majority_correct / conflict_majority_total
            if conflict_majority_total else None,
        },
        "all_key_majority_oracle": {
            "correct": sum(
                max(Counter(row["target"] for row in rows).values())
                for rows in buckets.values()
            ),
            "count": train_count,
            "accuracy": sum(
                max(Counter(row["target"] for row in rows).values())
                for rows in buckets.values()
            ) / train_count if train_count else None,
        },
        "conflicts": [summarize(key, rows) for key, rows in sorted(conflicts.items())],
        "interpretation": (
            "In-sample alias/label-consistency diagnostic only. The majority oracle is not a "
            "generalization score. Repeated text with conflicting targets can indicate that "
            "visual/state context is necessary, but does not prove those streams resolve it."
        ),
        "validation_metrics_computed": False,
        "sealed_audit_accessed": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.package_dir.resolve(strict=True) / "package-report.json"
    records_path = args.package_dir.resolve(strict=True) / "transition-examples.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("sealed_audit_accessed") is not False
            or report.get("action_columns_read") is not False
            or report.get("training_performed") is not False):
        raise ValueError("frozen package provenance flags do not pass")
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    if digest != report.get("transition_examples_sha256"):
        raise ValueError("transition examples hash does not match package report")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    result = audit_train_rows(payload["records"])
    result.update({
        "source_revision": payload["source_revision"],
        "package_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "transition_examples_sha256": digest,
        "rows_loaded": len(payload["records"]),
        "non_train_rows_used": 0,
    })
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "status", "train_examples", "unique_model_visible_text_keys",
            "repeated_text_keys", "conflicting_text_keys",
            "records_in_conflicting_text_keys", "conflicting_target_assignments_beyond_majority",
            "cross_episode_conflict_keys", "records_in_cross_episode_conflicts",
            "cross_video_conflict_keys", "records_in_cross_video_conflicts",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
