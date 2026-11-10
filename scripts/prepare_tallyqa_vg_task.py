"""Freeze a deterministic four-option adaptation of TallyQA AMT train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.screen_tallyqa_vg_train import load_training_rows, normalize_question, sha256_file


def options_for(answer: int) -> list[int]:
    if not 0 <= answer <= 15:
        raise ValueError("answer must be in [0, 15]")
    if answer == 0:
        return [0, 1, 2, 3]
    if answer == 1:
        return [1, 0, 2, 3]
    if answer == 15:
        return [15, 14, 13, 12]
    return [answer, answer - 1, answer + 1, answer - 2]


def option_order(question_id: int, values: list[int], seed: int = 17) -> list[int]:
    return sorted(
        values,
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
    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": 17,
        "source_repository": screen["source"]["repository"],
        "source_commit": screen["source"]["commit"],
        "source_archive_sha256": screen["archive_sha256"],
        "task": {
            "name": "tallyqa_vg_amt_four_option_counting",
            "description": (
                "Project-created four-option adaptation of question-conditioned open-ended "
                "count labels; not comparable to the official open-response benchmark."
            ),
            "option_policy": (
                "Answer plus nearest distinct counts; edge distractors 0:[1,2,3], "
                "1:[0,2,3], 15:[12,13,14]. Per-question deterministic SHA256 ordering."
            ),
            "controls": [
                "question_only",
                "joint_image_question",
                "joint_head_image_zero",
                "joint_head_image_shuffled",
            ],
            "official_test_opened": False,
        },
        "train": [],
        "validation": [],
    }
    rows_by_qid = {
        int(row["question_id"]): row for row in source_rows if row.get("data_source") == "amt"
    }
    for split in ("train", "validation"):
        allowed = split_ids[split]
        rows = []
        for qid in screen["split"][split]["question_ids"]:
            source = rows_by_qid[int(qid)]
            image_id = int(source["image_id"])
            receipt = receipts.get((split, image_id))
            if receipt is None or receipt["source_path"] != source["image"].replace("\\", "/"):
                raise ValueError("media receipt does not match source image identity")
            answer = int(source["answer"])
            ordered = option_order(int(qid), options_for(answer))
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
        if {row["question_id"] for row in rows} != allowed:
            raise ValueError("prepared rows do not match frozen split IDs")
        result[split] = rows
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
    result = prepare(screen, acquisition, load_training_rows(args.archive))
    result["source_screen_file_sha256"] = sha256_file(args.screen)
    result["acquisition_file_sha256"] = sha256_file(args.acquisition)
    result["source_archive_file_sha256"] = sha256_file(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train": len(result["train"]),
                "validation": len(result["validation"]),
                "task_manifest_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
