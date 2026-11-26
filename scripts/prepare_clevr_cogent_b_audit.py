"""Freeze one content-disjoint CoGenT-B audit question per official valB image."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tiny_omni_decision.clevr_vqa_dataset import (
    iter_question_array,
    normalize_question,
    select_cogent_b_audit_examples,
)

EXPECTED_ANNOTATION_SHA256 = "1b3a7de7624e4204cea9eb91c9125ebcebea8e92a9088606f6114ab39cedb9e5"
EXPECTED_VALB_IMAGES = 15_000
VALB_QUESTION_MEMBER = "CLEVR_CoGenT_v1.0/questions/CLEVR_valB_questions.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, expected_split: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"{path} must contain only non-empty {expected_split} rows")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    freeze = json.loads(args.selection_freeze.read_text(encoding="utf-8"))
    if freeze.get("selection_frozen_before_cogent_b_access") is not True:
        raise ValueError("candidate selection must be frozen before CoGenT-B access")
    if freeze.get("no_further_validation_tuning_after_freeze") is not True:
        raise ValueError("validation tuning must be frozen before CoGenT-B access")
    annotation_sha = sha256(args.annotations)
    if annotation_sha != EXPECTED_ANNOTATION_SHA256:
        raise ValueError(f"unexpected CoGenT annotation archive hash: {annotation_sha}")

    train = read_manifest(args.train_manifest, "train")
    validation = read_manifest(args.validation_manifest, "validation")
    development_rows = [*train, *validation]
    forbidden_images = {str(row["image_filename"]) for row in development_rows}
    if len(forbidden_images) != len(development_rows):
        raise ValueError("development manifests share image identities")
    excluded_questions = {
        normalize_question(str(row["question"])) for row in development_rows
    }

    started = datetime.now(UTC)
    with zipfile.ZipFile(args.annotations) as archive:
        selected = select_cogent_b_audit_examples(
            iter_question_array(archive, VALB_QUESTION_MEMBER),
            image_limit=EXPECTED_VALB_IMAGES,
            seed=args.seed,
            excluded_normalized_questions=excluded_questions,
            forbidden_image_filenames=forbidden_images,
        )

    audit_questions = {normalize_question(str(row["question"])) for row in selected}
    if audit_questions & excluded_questions:
        raise ValueError("normalized question content overlaps a development split")
    if {row["image_filename"] for row in selected} & forbidden_images:
        raise ValueError("audit image identity overlaps a development split")
    if args.seed != 17:
        raise ValueError("this frozen candidate audit is specified with seed 17")

    args.output.mkdir(parents=True)
    manifest_path = args.output / "audit-examples.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    metadata = {
        "status": "FROZEN_COGENT_B_AUDIT_MANIFEST",
        "source": "official CLEVR-CoGenT v1.0 valB question annotations",
        "source_version_id": "1pgXk1N966vKCADkX4VK8BtdpNFyzYXD",
        "source_etag": '"1900bbeda991856079316bcc7c464138-8"',
        "source_archive_sha256": annotation_sha,
        "source_question_member": VALB_QUESTION_MEMBER,
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "selection_started_at_utc": started.isoformat(),
        "candidate_selection_freeze_sha256": sha256(args.selection_freeze),
        "seed": args.seed,
        "policy": (
            "one hash-ranked question per each of all 15,000 valB images; exclude exact normalized "
            "question text seen in train/validation; fail closed rather than shrink image coverage"
        ),
        "audit_examples": len(selected),
        "unique_images": len({row["image_filename"] for row in selected}),
        "unique_source_questions": len({row["sample_id"] for row in selected}),
        "train_manifest_sha256": sha256(args.train_manifest),
        "validation_manifest_sha256": sha256(args.validation_manifest),
        "manifest_sha256": sha256(manifest_path),
        "train_image_overlap": 0,
        "validation_image_overlap": 0,
        "normalized_question_overlap_with_development": 0,
        "answer_kind_counts": dict(Counter(str(row["answer_kind"]) for row in selected)),
        "outer_program_counts": dict(Counter(str(row["outer_program"]) for row in selected)),
        "option_count_counts": dict(Counter(str(len(row["options"])) for row in selected)),
        "choice_policy": "same-answer-kind choices from make_options, seed 17",
        "candidate_reselection_after_audit": False,
        "audit_metrics_computed": False,
        "CoGenT_testA_or_testB_opened": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
