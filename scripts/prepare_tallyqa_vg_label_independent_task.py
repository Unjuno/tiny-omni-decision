"""Freeze a TallyQA adaptation whose count options do not depend on the label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.screen_tallyqa_vg_train import load_training_rows, normalize_question, sha256_file

SEED = 17
COUNT_OPTIONS = tuple(range(16))
TASK_ID = "tallyqa_vg_amt_16_choice_label_independent_counting"
PINNED_SOURCE_COMMIT = "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f"
PINNED_ARCHIVE_SHA256 = "e805199fa25c91d2f9df964920c6c36ce8d6cb941c970176f38a92f971bab5ea"
PINNED_SCREEN_SHA256 = "110fbbe4983106dceb90e82d5454aa40dd3eaebd6151fae03c8f6e6af40a516a"
PINNED_ACQUISITION_SHA256 = "13707d624dcd3d9ee814b85436d5346f230845a095a5b8067bb15b396976f952"


def ordered_options(question_id: int, seed: int = SEED) -> list[int]:
    """Return all supported counts in a stable order independent of the answer."""
    return sorted(
        COUNT_OPTIONS,
        key=lambda value: hashlib.sha256(f"{seed}:{question_id}:{value}".encode("ascii")).digest(),
    )


def prepare(
    screen: dict[str, Any], acquisition: dict[str, Any], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if acquisition.get("status") != "complete":
        raise ValueError("complete media acquisition is required")
    source_qids = {
        int(row["question_id"]) for row in source_rows if row.get("data_source") == "amt"
    }
    split_ids = {
        split: set(map(int, screen["split"][split]["question_ids"]))
        for split in ("train", "validation")
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or source_qids != split_ids["train"] | split_ids["validation"]
    ):
        raise ValueError("split IDs do not match source rows or overlap")
    receipts = {
        (item["split"], int(item["image_id"])): item
        for key in ("train_receipts", "validation_receipts")
        for item in acquisition[key]
    }
    rows_by_qid = {
        int(row["question_id"]): row for row in source_rows if row.get("data_source") == "amt"
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": SEED,
        "source_repository": screen["source"]["repository"],
        "source_commit": screen["source"]["commit"],
        "source_archive_sha256": screen["archive_sha256"],
        "task": {
            "id": TASK_ID,
            "name": "question-conditioned Visual Genome count selection, 16 choices",
            "description": (
                "Project-created fixed-vocabulary 16-choice adaptation of TallyQA AMT rows. "
                "The complete candidate set is independent of the answer; not comparable to "
                "the official open-response benchmark."
            ),
            "option_policy": (
                "All integer counts 0..15 are supplied for every question. Order is a stable "
                "SHA-256 permutation of seed and question ID, independent of the answer."
            ),
            "option_count": len(COUNT_OPTIONS),
            "controls": [
                "question_only",
                "joint_image_question",
                "joint_head_image_zero",
                "joint_head_image_shuffled",
            ],
            "official_test_opened": False,
            "validation_prior_exposure": (
                "The source validation rows were used by the prior four-choice development "
                "probe; this task manifest is not a blind evaluation set."
            ),
        },
        "train": [],
        "validation": [],
    }
    for split in ("train", "validation"):
        rows = []
        for qid in screen["split"][split]["question_ids"]:
            source = rows_by_qid[int(qid)]
            image_id = int(source["image_id"])
            receipt = receipts.get((split, image_id))
            if receipt is None or receipt["source_path"] != source["image"].replace("\\", "/"):
                raise ValueError("media receipt does not match source image identity")
            answer = int(source["answer"])
            ordered = ordered_options(int(qid))
            if answer not in COUNT_OPTIONS:
                raise ValueError("source answer falls outside the fixed count vocabulary")
            rows.append(
                {
                    "question_id": int(qid),
                    "image_id": image_id,
                    "question": source["question"],
                    "normalized_question": normalize_question(source["question"]),
                    "answer": answer,
                    "options": [
                        {"id": f"count_{value}", "text": str(value), "value": value}
                        for value in ordered
                    ],
                    "target": ordered.index(answer),
                    "media_sha256": receipt["sha256"],
                }
            )
        if {row["question_id"] for row in rows} != split_ids[split]:
            raise ValueError("prepared rows do not match frozen split IDs")
        result[split] = rows
    train_rows, validation_rows = result["train"], result["validation"]
    if (
        {row["image_id"] for row in train_rows}
        & {row["image_id"] for row in validation_rows}
        or {row["normalized_question"] for row in train_rows}
        & {row["normalized_question"] for row in validation_rows}
        or {row["media_sha256"] for row in train_rows}
        & {row["media_sha256"] for row in validation_rows}
    ):
        raise ValueError("image, normalized-question, or media identity overlaps across splits")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    checked_inputs = {
        "source_commit": screen["source"]["commit"] == PINNED_SOURCE_COMMIT,
        "screen_archive_sha256": screen["archive_sha256"] == PINNED_ARCHIVE_SHA256,
        "screen_file_sha256": {
            "path": str(args.screen),
            "actual": sha256_file(args.screen),
            "expected": PINNED_SCREEN_SHA256,
        },
        "acquisition_file_sha256": {
            "path": str(args.acquisition),
            "actual": sha256_file(args.acquisition),
            "expected": PINNED_ACQUISITION_SHA256,
        },
        "annotation_archive_sha256": {
            "path": str(args.archive),
            "actual": sha256_file(args.archive),
            "expected": PINNED_ARCHIVE_SHA256,
        },
    }
    mismatches = {
        key: value
        for key, value in checked_inputs.items()
        if value is False or (isinstance(value, dict) and value["actual"] != value["expected"])
    }
    if mismatches:
        raise ValueError(f"pinned-input mismatch: {mismatches}")
    result = prepare(screen, acquisition, load_training_rows(args.archive))
    result["source_screen_file_sha256"] = sha256_file(args.screen)
    result["acquisition_file_sha256"] = sha256_file(args.acquisition)
    result["source_archive_file_sha256"] = sha256_file(args.archive)
    result["source_legacy_task_manifest_sha256"] = (
        "59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "train": len(result["train"]),
                "validation": len(result["validation"]),
                "task_manifest_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
