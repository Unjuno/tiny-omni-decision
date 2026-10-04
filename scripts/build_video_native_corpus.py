from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.dataset import iter_local_rows, sha256_file  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import DatasetManifest, DecisionExample  # noqa: E402
from tiny_omni_decision.video_corpus import (  # noqa: E402
    VIDEO_TASK_TYPES,
    build_video_native_corpora,
)

TASK_WEIGHTS = {
    "temporal_descriptive": 0.2,
    "explanatory": 0.3,
    "predictive": 0.3,
    "counterfactual": 0.2,
}


def _read_examples(path: Path) -> list[DecisionExample]:
    with path.open(encoding="utf-8") as handle:
        return [
            DecisionExample.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _write_examples(path: Path, examples: list[DecisionExample]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.model_dump_json(exclude_none=True) + "\n")


def _load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(load_structured_file(path))


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "base_train": args.base_train,
        "base_validation": args.base_validation,
        "raw_train_questions": args.raw_train_questions,
        "raw_validation_questions": args.raw_validation_questions,
        "train_source_manifest": args.train_source_manifest,
        "validation_source_manifest": args.validation_source_manifest,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    output_dir = args.output_dir.resolve()
    artifact_root = (ROOT / "artifacts" / "tiny-omni-decision-teacher-v2").resolve()
    if not output_dir.is_relative_to(artifact_root):
        raise ValueError("output directory must stay under artifacts/tiny-omni-decision-teacher-v2")
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; refusing to overwrite: " f"{output_dir}"
        )

    base_train = _read_examples(args.base_train)
    base_validation = _read_examples(args.base_validation)
    train_manifest = _load_manifest(args.train_source_manifest)
    validation_manifest = _load_manifest(args.validation_source_manifest)
    train, validation, report = build_video_native_corpora(
        base_train,
        base_validation,
        iter_local_rows(args.raw_train_questions),
        iter_local_rows(args.raw_validation_questions),
        train_manifest,
        validation_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_examples(train_path, train)
    _write_examples(validation_path, validation)

    manifest = {
        "schema_version": 1,
        "corpus_id": "tiny-omni-decision-teacher-v2-candidate-e",
        "purpose": "validation-selected video-native task-mix experiment",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "build_argv": [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        "source_revision": train_manifest.revision,
        "base_teacher_id": "tiny-omni-decision-teacher-v1",
        "input_sha256": {
            label: sha256_file(path) for label, path in paths.items()
        },
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "train_path": train_path.name,
        "validation_path": validation_path.name,
        "video_question_sampling_weights": TASK_WEIGHTS,
        "video_task_types": list(VIDEO_TASK_TYPES),
        "video_media_reference_policy": (
            "reuse the locally materialized Teacher v1 video reference for each frozen scene"
        ),
        "video_question_format": {
            "descriptive": (
                "original categorical DecisionExample retained if temporally descriptive"
            ),
            "native_multiple_choice": (
                "one binary [wrong, correct] DecisionExample per source choice"
            ),
            "candidate_prompt_suffix": "Candidate statement: <source choice text>",
            "parent_question_id": "scene_index:question_id",
        },
        "static_descriptive_questions_in_candidate": False,
        "sealed_audit_loaded": False,
        "report": report,
    }
    manifest_path = output_dir / "corpus-manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate CLEVRER video-native corpus from the frozen Teacher v1 "
            "train/validation video identities."
        )
    )
    parser.add_argument(
        "--base-train",
        type=Path,
        default=ROOT / "data/processed/durable-teacher-v1/train.jsonl",
    )
    parser.add_argument(
        "--base-validation",
        type=Path,
        default=ROOT / "data/processed/durable-teacher-v1/validation.jsonl",
    )
    parser.add_argument(
        "--raw-train-questions",
        type=Path,
        default=ROOT / "data/raw/clevrer/train-questions.json",
    )
    parser.add_argument(
        "--raw-validation-questions",
        type=Path,
        default=ROOT / "data/raw/clevrer/validation-questions.json",
    )
    parser.add_argument(
        "--train-source-manifest",
        type=Path,
        default=ROOT / "manifests/candidates/clevrer-video-native.yaml",
    )
    parser.add_argument(
        "--validation-source-manifest",
        type=Path,
        default=ROOT / "manifests/candidates/clevrer-video-native-validation.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/tiny-omni-decision-teacher-v2/candidate-e/corpus-v1",
    )
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
