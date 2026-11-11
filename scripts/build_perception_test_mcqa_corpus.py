"""Freeze a video-grouped Perception Test MCQA corpus from train/valid only."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.audit_perception_test_mcqa_exposure import _normalized_question_content
except ModuleNotFoundError:
    from audit_perception_test_mcqa_exposure import _normalized_question_content

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
CONTENT_NORMALIZATION = (
    "Unicode NFKC, casefold, collapsed whitespace, order-insensitive options; "
    "SHA-256 of question plus three options, excluding answer labels."
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_split(archive_path: str | Path, split: str) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("only train and valid annotation archives are supported")
    with zipfile.ZipFile(archive_path) as archive:
        member = f"all_{split}.json"
        if member not in archive.namelist():
            raise ValueError(f"annotation archive is missing {member}")
        rows = json.loads(archive.read(member))
    if not isinstance(rows, dict):
        raise ValueError(f"{member} must contain an object keyed by video ID")
    for video_id, video in rows.items():
        if not isinstance(video, dict):
            raise ValueError(f"video annotation must be an object: {video_id}")
        metadata = video.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("split") != split
            or metadata.get("video_id") != video_id
        ):
            raise ValueError(f"split/video metadata mismatch: {split}/{video_id}")
        if not isinstance(video.get("mc_question"), list):
            raise ValueError(f"MCQA list is missing: {split}/{video_id}")
    return rows


def _decision_record(split: str, video_id: str, question: dict[str, Any]) -> dict[str, Any]:
    question_id = question.get("id")
    prompt = question.get("question")
    options = question.get("options")
    answer_id = question.get("answer_id")
    if isinstance(question_id, bool) or not isinstance(question_id, int):
        raise ValueError(f"invalid question ID: {split}/{video_id}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"question text is missing: {split}/{video_id}/{question_id}")
    if (
        not isinstance(options, list)
        or len(options) != 3
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError(f"expected three nonempty options: {split}/{video_id}/{question_id}")
    if (
        isinstance(answer_id, bool)
        or not isinstance(answer_id, int)
        or not 0 <= answer_id < len(options)
    ):
        raise ValueError(f"invalid answer index: {split}/{video_id}/{question_id}")
    labeled_options = [
        f"{chr(ord('A') + index)}: {option.strip()}" for index, option in enumerate(options)
    ]
    return {
        "id": f"perception-test/{split}/{video_id}/{question_id}",
        "split": split,
        "group_id": video_id,
        "source_video_id": video_id,
        "source_record_id": f"{video_id}/{question_id}",
        "question_id": question_id,
        "question": prompt.strip(),
        "options": labeled_options,
        "target_option_index": answer_id,
        "target": labeled_options[answer_id],
        "reasoning_type": str(question.get("reasoning", "unknown")),
        "area": str(question.get("area", "unknown")),
        "normalized_content_sha256": _normalized_question_content(question),
        "source": "google-deepmind/perception_test",
        "source_revision": SOURCE_REVISION,
        "license": "CC-BY-4.0",
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _hash_order(rows: list[dict[str, Any]], key: str) -> str:
    return hashlib.sha256("\n".join(str(row[key]) for row in rows).encode("utf-8")).hexdigest()


def build_corpus(
    *,
    train_annotations: str | Path,
    valid_annotations: str | Path,
    exposure_report: dict[str, Any],
    seed: int = 17,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_sha = sha256_file(train_annotations)
    valid_sha = sha256_file(valid_annotations)
    expected_train_sha = exposure_report["official_train"]["annotation_archive_sha256"]
    expected_valid_sha = exposure_report["official_valid"]["annotation_archive_sha256"]
    if train_sha != expected_train_sha or valid_sha != expected_valid_sha:
        raise ValueError("annotation archive hash differs from exposure-audit inputs")

    train_videos = _read_split(train_annotations, "train")
    valid_videos = _read_split(valid_annotations, "valid")
    overlap = set(train_videos) & set(valid_videos)
    if overlap:
        raise ValueError(f"train/valid video overlap: {sorted(overlap)[:3]}")

    candidate_video_ids = set(
        exposure_report["validation_candidates_after_train_content_deduplication"][
            "video_ids"
        ]
    )
    if not candidate_video_ids.issubset(valid_videos):
        raise ValueError("exposure report includes unknown validation video IDs")

    train_rows: list[dict[str, Any]] = []
    train_content: set[str] = set()
    for video_id in sorted(train_videos):
        questions = sorted(train_videos[video_id]["mc_question"], key=lambda row: row["id"])
        for question in questions:
            row = _decision_record("train", video_id, question)
            train_content.add(row["normalized_content_sha256"])
            train_rows.append(row)

    valid_rows: list[dict[str, Any]] = []
    valid_videos_with_questions: set[str] = set()
    for video_id in sorted(candidate_video_ids):
        questions = sorted(valid_videos[video_id]["mc_question"], key=lambda row: row["id"])
        for question in questions:
            row = _decision_record("valid", video_id, question)
            if row["normalized_content_sha256"] in train_content:
                continue
            valid_rows.append(row)
            valid_videos_with_questions.add(video_id)
    if valid_videos_with_questions != candidate_video_ids:
        raise ValueError("candidate video group has no train-content-disjoint MCQA row")
    if {row["normalized_content_sha256"] for row in train_rows} & {
        row["normalized_content_sha256"] for row in valid_rows
    }:
        raise ValueError("normalized question/choice content overlaps across splits")

    type_counts: Counter[str] = Counter(row["reasoning_type"] for row in valid_rows)
    type_groups: dict[str, set[str]] = {}
    for row in valid_rows:
        type_groups.setdefault(row["reasoning_type"], set()).add(row["group_id"])
    train_bytes = _jsonl_bytes(train_rows)
    valid_bytes = _jsonl_bytes(valid_rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": "r2-perception-test-mcqa-path-a-20261111",
        "seed": seed,
        "source": {
            "dataset_id": "google-deepmind/perception_test",
            "revision": SOURCE_REVISION,
            "license": "CC-BY-4.0",
            "train_annotation_archive_sha256": train_sha,
            "valid_annotation_archive_sha256": valid_sha,
            "official_test_opened": False,
        },
        "exposure_audit": {
            "protocol": exposure_report["protocol"],
            "validation_video_id_list_sha256": hashlib.sha256(
                "\n".join(sorted(candidate_video_ids)).encode("utf-8")
            ).hexdigest(),
            "validation_group_count": len(candidate_video_ids),
            "direct_artifact_match_absence_is_not_proof_of_non_exposure": True,
        },
        "split_group": "source_video_id",
        "train": {
            "record_count": len(train_rows),
            "video_group_count": len({row["group_id"] for row in train_rows}),
            "record_order_sha256": _hash_order(train_rows, "id"),
            "records_sha256": hashlib.sha256(train_bytes).hexdigest(),
        },
        "validation": {
            "record_count": len(valid_rows),
            "video_group_count": len(valid_videos_with_questions),
            "record_order_sha256": _hash_order(valid_rows, "id"),
            "records_sha256": hashlib.sha256(valid_bytes).hexdigest(),
            "normalized_content_overlap_with_train": 0,
            "reasoning_type_counts": {
                reasoning: {
                    "questions": type_counts[reasoning],
                    "video_groups": len(type_groups[reasoning]),
                }
                for reasoning in sorted(type_counts)
            },
        },
        "group_integrity": {
            "train_validation_video_id_overlap": 0,
            "all_questions_for_a_video_remain_in_one_split": True,
        },
        "normalization": CONTENT_NORMALIZATION,
        "data_policy": {
            "validation_rows_with_exact_train_question_choice_content_removed": True,
            "participant_disjointness_proven": False,
            "unseen_script_generalization_proven": False,
            "media_identity_gate_pending": True,
            "model_fitting_or_scoring_performed": False,
        },
    }
    return train_rows, valid_rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--valid-annotations", type=Path, required=True)
    parser.add_argument("--exposure-report", type=Path, required=True)
    parser.add_argument("--expected-exposure-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen corpus directory: {args.output}")
    report_sha = sha256_file(args.exposure_report)
    if report_sha != args.expected_exposure_report_sha256:
        raise ValueError("exposure report hash mismatch")
    exposure_report = json.loads(args.exposure_report.read_text(encoding="utf-8"))
    train_rows, valid_rows, manifest = build_corpus(
        train_annotations=args.train_annotations,
        valid_annotations=args.valid_annotations,
        exposure_report=exposure_report,
        seed=args.seed,
    )
    manifest["exposure_audit"]["report_sha256"] = report_sha
    args.output.mkdir(parents=True, exist_ok=False)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "validation.jsonl"
    manifest_path = args.output / "manifest.json"
    train_path.write_bytes(_jsonl_bytes(train_rows))
    valid_path.write_bytes(_jsonl_bytes(valid_rows))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_CANDIDATE_CORPUS",
                "train_records": len(train_rows),
                "train_video_groups": manifest["train"]["video_group_count"],
                "validation_records": len(valid_rows),
                "validation_video_groups": len(valid_videos_with_questions(valid_rows)),
                "manifest_sha256": sha256_file(manifest_path),
                "model_fitting_or_scoring_performed": False,
            },
            indent=2,
        ),
        flush=True,
    )


def valid_videos_with_questions(rows: list[dict[str, Any]]) -> set[str]:
    return {row["group_id"] for row in rows}


if __name__ == "__main__":
    main()
