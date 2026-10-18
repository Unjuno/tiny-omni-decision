"""Freeze a scene-disjoint CLEVRER temporal task-mix split for development."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.fetch_clevrer_probe import REVISION, TRAIN_QUESTIONS_SHA256
from scripts.prepare_clevrer_program_order_sample import discover_previously_observed_scenes
from scripts.select_clevrer_temporal_split import TEMPORAL_TASK_TYPES, select_temporal_task_mix

SEED = 17
VALIDATION_SCENES_PER_TYPE = 200
TRAIN_EXAMPLE_COUNTS = {
    "temporal_descriptive": 204,
    "explanatory": 306,
    "predictive": 306,
    "counterfactual": 208,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()
    path.write_bytes(payload)
    return _sha256(payload)


def prepare_split(
    *, source_questions: Path, artifacts_root: Path, output_dir: Path, seed: int = SEED
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite temporal split: {output_dir}")
    source_bytes = source_questions.read_bytes()
    source_hash = _sha256(source_bytes)
    if source_hash != TRAIN_QUESTIONS_SHA256:
        raise ValueError("CLEVRER train questions do not match the pinned source revision")
    raw_rows = json.loads(source_bytes)
    excluded_scenes, exclusion_sources = discover_previously_observed_scenes(artifacts_root)
    train, validation = select_temporal_task_mix(
        raw_rows,
        validation_scenes_per_type=VALIDATION_SCENES_PER_TYPE,
        train_example_counts=TRAIN_EXAMPLE_COUNTS,
        seed=seed,
        excluded_scenes=excluded_scenes,
    )
    train_scenes = {int(row["scene_index"]) for row in train}
    validation_scenes = {int(row["scene_index"]) for row in validation}
    if train_scenes & validation_scenes or (train_scenes | validation_scenes) & excluded_scenes:
        raise ValueError("selected CLEVRER scenes overlap or include previously observed scenes")

    output_dir.mkdir(parents=True, exist_ok=False)
    split_hashes = {
        "train.jsonl": _write_jsonl(output_dir / "train.jsonl", train),
        "validation.jsonl": _write_jsonl(output_dir / "validation.jsonl", validation),
    }
    excluded_bytes = (
        json.dumps(sorted(excluded_scenes), separators=(",", ":")) + "\n"
    ).encode()
    (output_dir / "excluded-scenes.json").write_bytes(excluded_bytes)
    id_order_hashes = {
        "train": _sha256("\n".join(row["id"] for row in train).encode()),
        "validation": _sha256("\n".join(row["id"] for row in validation).encode()),
    }
    normalized_content = {
        split: {
            (
                " ".join(str(row["question"]).casefold().split()),
                tuple(
                    sorted(
                        " ".join(str(option).casefold().split())
                        for option in row["options"]
                    )
                ),
            )
            for row in rows
        }
        for split, rows in (("train", train), ("validation", validation))
    }
    task_rows = {
        split: dict(Counter(str(row["question_type"]) for row in rows))
        for split, rows in (("train", train), ("validation", validation))
    }
    target_counts = {
        split: {
            f"{task}/{target}": count
            for (task, target), count in Counter(
                (str(row["question_type"]), str(row["target"])) for row in rows
            ).items()
        }
        for split, rows in (("train", train), ("validation", validation))
    }
    report = {
        "dataset": "MIT-IBM/CLEVRER",
        "revision": REVISION,
        "license": "CC0-1.0",
        "source_questions_sha256": source_hash,
        "seed": seed,
        "selector": (
            "scene-first 20/30/30/20 temporal mix; reasoning parents yield one correct and "
            "one wrong binary candidate; normalized content and scene identity are disjoint"
        ),
        "question_types": list(TEMPORAL_TASK_TYPES),
        "selection": {
            "train_example_counts": TRAIN_EXAMPLE_COUNTS,
            "validation_scenes_per_type": VALIDATION_SCENES_PER_TYPE,
        },
        "counts": {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_unique_scenes": len(train_scenes),
            "validation_unique_scenes": len(validation_scenes),
            "train_unique_parent_questions": len({row["parent_question_id"] for row in train}),
            "validation_unique_parent_questions": len(
                {row["parent_question_id"] for row in validation}
            ),
            "train_unique_normalized_content": len(normalized_content["train"]),
            "validation_unique_normalized_content": len(normalized_content["validation"]),
            "train_question_type_rows": task_rows["train"],
            "validation_question_type_rows": task_rows["validation"],
            "train_target_counts": target_counts["train"],
            "validation_target_counts": target_counts["validation"],
            "prior_observed_scene_count": len(excluded_scenes),
            "excluded_scene_ids_sha256": _sha256(excluded_bytes),
            "exclusion_source_count": len(exclusion_sources),
            "exclusion_sources_sha256": _sha256(
                json.dumps(exclusion_sources, sort_keys=True).encode()
            ),
            "selected_scenes_previously_observed": 0,
            "scene_overlap": 0,
            "parent_question_overlap": 0,
            "normalized_question_options_overlap": 0,
            "train_id_order_sha256": id_order_hashes["train"],
            "validation_id_order_sha256": id_order_hashes["validation"],
            "split_file_sha256": split_hashes,
            "excluded_scenes_file_sha256": _sha256(excluded_bytes),
        },
        "media_extracted": False,
        "sealed_audit_loaded": False,
        "exclusion_sources": exclusion_sources,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-questions", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_split(
                source_questions=args.source_questions,
                artifacts_root=args.artifacts_root,
                output_dir=args.output_dir,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
