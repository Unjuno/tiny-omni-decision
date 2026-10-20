"""Freeze a scene-disjoint text-free temporal CLEVRER coverage corpus."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tiny_omni_decision.clevrer import TAXONOMIES, classify

DATASET_REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
SOURCE_ID = f"MIT-IBM/CLEVRER@{DATASET_REVISION}"
TRAIN_QUESTIONS_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
CANDIDATE_SCENE_IDS_SHA256 = "33bb4408f20ec95a64ed657da5e860588dccfd5378b95c52101ab0768baba4e8"
ANNOTATION_RECEIPT_SHA256 = "26c178bbfcbc06f38828d47592e7cce7a545c0372f4f9b7f676e7322a6e41485"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(scene_ids: set[int]) -> str:
    return sha256_bytes(",".join(map(str, sorted(scene_ids))).encode("ascii"))


def _scene_order(scene_ids: set[int], seed: int) -> list[int]:
    return sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(
            f"clevrer-textfree-temporal|{seed}|scene|{scene_id}".encode()
        ).hexdigest(),
    )


def _content_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(value).casefold().split()) for value in row["options"]))
    return question, options


def build_temporal_coverage_rows(
    raw_rows: list[dict[str, Any]],
    candidate_scene_ids: set[int],
    *,
    seed: int = 17,
    train_fraction: float = 0.8,
) -> dict[str, Any]:
    """Build one all-temporal-question row set, grouped and split by video scene.

    The natural-language question is retained for split auditing and provenance,
    but the text-free readout consumes only video features, program tokens,
    taxonomy IDs, and supplied options.
    """
    if not candidate_scene_ids or not 0.0 < train_fraction < 1.0:
        raise ValueError("candidate scenes and a train fraction in (0,1) are required")
    by_scene: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        scene_id = raw.get("scene_index")
        if not isinstance(scene_id, int) or isinstance(scene_id, bool):
            raise ValueError("CLEVRER scene_index must be an integer")
        if scene_id in candidate_scene_ids:
            if scene_id in by_scene:
                raise ValueError(f"duplicate source row for candidate scene {scene_id}")
            by_scene[scene_id] = raw
    missing_scenes = candidate_scene_ids - set(by_scene)
    if missing_scenes:
        raise ValueError(
            f"candidate scene IDs are missing from questions: {sorted(missing_scenes)}"
        )

    ordered_scenes = _scene_order(candidate_scene_ids, seed)
    train_count = int(len(ordered_scenes) * train_fraction)
    if train_count < 1 or train_count >= len(ordered_scenes):
        raise ValueError("scene partition would leave an empty split")
    train_scenes = set(ordered_scenes[:train_count])
    validation_scenes = set(ordered_scenes[train_count:])

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for scene_id in sorted(candidate_scene_ids):
        raw = by_scene[scene_id]
        video_filename = raw.get("video_filename")
        if not isinstance(video_filename, str) or not video_filename:
            raise ValueError(f"candidate scene {scene_id} has no video filename")
        split = "train" if scene_id in train_scenes else "validation"
        for question in raw.get("questions", []):
            if question.get("question_type") != "descriptive":
                continue
            if classify(question) != "temporal_descriptive":
                continue
            taxonomy = question.get("question_subtype")
            answer = question.get("answer")
            program = question.get("program")
            question_text = question.get("question")
            question_id = question.get("question_id")
            if taxonomy not in TAXONOMIES or answer not in TAXONOMIES[taxonomy]:
                continue
            if (
                not isinstance(question_text, str)
                or not isinstance(program, list)
                or not program
                or not isinstance(question_id, int)
                or isinstance(question_id, bool)
            ):
                raise ValueError(f"invalid temporal descriptive question in scene {scene_id}")
            row_id = f"clevrer-temporal-{scene_id:05d}-q{question_id:03d}"
            rows_by_split[split].append(
                {
                    "id": row_id,
                    "split": split,
                    "source": SOURCE_ID,
                    "scene_index": scene_id,
                    "scene_group_id": f"clevrer-train-scene-{scene_id:05d}",
                    "video_filename": video_filename,
                    "question_id": question_id,
                    "question_type": "temporal_descriptive",
                    "taxonomy": str(taxonomy),
                    "question": question_text,
                    "options": list(TAXONOMIES[taxonomy]),
                    "target": str(answer),
                    "program": [str(token) for token in program],
                    "media_path": f"media/{video_filename}",
                }
            )

    scenes_with_temporal_rows = {
        int(row["scene_index"]) for rows in rows_by_split.values() for row in rows
    }
    missing_temporal_scenes = candidate_scene_ids - scenes_with_temporal_rows
    if missing_temporal_scenes:
        raise ValueError(
            f"candidate scene has no temporal descriptive rows: {sorted(missing_temporal_scenes)}"
        )

    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"{split} has no temporal descriptive rows")
        scenes_with_rows = {int(row["scene_index"]) for row in rows}
        expected_scenes = train_scenes if split == "train" else validation_scenes
        missing = expected_scenes - scenes_with_rows
        if missing:
            raise ValueError(f"candidate scene has no temporal descriptive rows: {sorted(missing)}")
        rows.sort(key=lambda row: (int(row["scene_index"]), int(row["question_id"])))
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate sample IDs in {split}")

    train_content = {_content_key(row) for row in rows_by_split["train"]}
    validation_before = rows_by_split["validation"]
    rows_by_split["validation"] = [
        row for row in validation_before if _content_key(row) not in train_content
    ]
    validation_content = {_content_key(row) for row in rows_by_split["validation"]}
    if train_content & validation_content:
        raise ValueError("normalized question/options overlap remains across splits")
    surviving_validation_scenes = {int(row["scene_index"]) for row in rows_by_split["validation"]}
    missing_validation_scenes = validation_scenes - surviving_validation_scenes
    if missing_validation_scenes:
        raise ValueError(
            "content de-duplication removed all validation rows for scenes: "
            f"{sorted(missing_validation_scenes)}"
        )

    report = {
        "seed": seed,
        "train_fraction": train_fraction,
        "candidate_scene_count": len(candidate_scene_ids),
        "candidate_scene_ids_sha256": _ids_sha256(candidate_scene_ids),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "train_scene_ids_sha256": _ids_sha256(train_scenes),
        "validation_scene_ids_sha256": _ids_sha256(validation_scenes),
        "train_rows": len(rows_by_split["train"]),
        "validation_rows_before_content_filter": len(validation_before),
        "validation_rows_dropped_for_content_overlap": len(validation_before)
        - len(rows_by_split["validation"]),
        "validation_rows": len(rows_by_split["validation"]),
        "train_question_type_counts": dict(
            sorted(Counter(str(row["question_type"]) for row in rows_by_split["train"]).items())
        ),
        "validation_question_type_counts": dict(
            sorted(
                Counter(str(row["question_type"]) for row in rows_by_split["validation"]).items()
            )
        ),
        "train_taxonomy_counts": dict(
            sorted(Counter(str(row["taxonomy"]) for row in rows_by_split["train"]).items())
        ),
        "validation_taxonomy_counts": dict(
            sorted(Counter(str(row["taxonomy"]) for row in rows_by_split["validation"]).items())
        ),
        "sample_id_order_sha256": {
            split: sha256_bytes("\n".join(str(row["id"]) for row in rows).encode("utf-8"))
            for split, rows in rows_by_split.items()
        },
        "normalized_question_option_overlap": 0,
        "natural_language_question_in_model_input": False,
        "model_input_fields": ["video_features", "program", "taxonomy", "options"],
        "validation_role": "development-only; group-disjoint within CLEVRER official train split",
    }
    return {
        "train_rows": rows_by_split["train"],
        "validation_rows": rows_by_split["validation"],
        "report": report,
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--exposure-receipt", type=Path, required=True)
    parser.add_argument("--annotation-receipt", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    questions_bytes = args.questions.read_bytes()
    if sha256_bytes(questions_bytes) != TRAIN_QUESTIONS_SHA256:
        raise ValueError("CLEVRER train questions do not match the pinned source revision")
    exposure_bytes = args.exposure_receipt.read_bytes()
    exposure = json.loads(exposure_bytes)
    candidate_scene_ids = {int(value) for value in exposure["candidate_scene_ids"]}
    if (
        len(candidate_scene_ids) != exposure.get("candidate_scene_count")
        or _ids_sha256(candidate_scene_ids) != CANDIDATE_SCENE_IDS_SHA256
        or exposure.get("candidate_scene_ids_sha256") != CANDIDATE_SCENE_IDS_SHA256
    ):
        raise ValueError("candidate scene receipt does not match the pinned exposure audit")
    annotation_bytes = args.annotation_receipt.read_bytes()
    if sha256_bytes(annotation_bytes) != ANNOTATION_RECEIPT_SHA256:
        raise ValueError("candidate annotation receipt hash mismatch")
    annotation_receipt = json.loads(annotation_bytes)
    annotation_ids = {int(row["scene_index"]) for row in annotation_receipt["entries"]}
    if annotation_ids != candidate_scene_ids:
        raise ValueError(
            "candidate annotation receipt does not cover the exact candidate scene set"
        )
    for row in annotation_receipt["entries"]:
        annotation_path = args.annotation_dir / f"annotation_{int(row['scene_index']):05d}.json"
        if sha256_bytes(annotation_path.read_bytes()) != row["sha256"]:
            raise ValueError(f"annotation hash mismatch for scene {row['scene_index']}")
    raw_rows = json.loads(questions_bytes)
    result = build_temporal_coverage_rows(raw_rows, candidate_scene_ids)
    report = {
        **result["report"],
        "source": {
            "dataset": "MIT-IBM/CLEVRER",
            "revision": DATASET_REVISION,
            "license": "CC0-1.0",
            "questions_sha256": TRAIN_QUESTIONS_SHA256,
            "exposure_receipt_sha256": sha256_bytes(exposure_bytes),
            "annotation_receipt_sha256": sha256_bytes(annotation_bytes),
        },
        "media_acquired": False,
        "media_sha256_verified": False,
        "feature_cache_status": "pending_media_acquisition",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "train.jsonl", result["train_rows"])
    _write_jsonl(output / "validation.jsonl", result["validation_rows"])
    _write_json(output / "corpus-preflight.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "train_rows": report["train_rows"],
                "validation_rows": report["validation_rows"],
                "train_scenes": report["train_scene_count"],
                "validation_scenes": report["validation_scene_count"],
                "sample_id_order_sha256": report["sample_id_order_sha256"],
                "media_acquired": report["media_acquired"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
