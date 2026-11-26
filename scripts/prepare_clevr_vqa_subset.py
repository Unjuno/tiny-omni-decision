"""Create a deterministic image-grouped CLEVR train/development subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from tiny_omni_decision.clevr_vqa_dataset import (
    ANNOTATION_MEMBERS,
    iter_question_array,
    normalize_question,
    require_group_disjoint,
    select_one_question_per_image,
)

EXPECTED_ARCHIVE_SHA256 = "d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=16384)
    parser.add_argument("--validation-images", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    archive_sha = sha256(args.annotations)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"unexpected CLEVR annotation archive SHA-256: {archive_sha}")
    with zipfile.ZipFile(args.annotations) as archive:
        train = select_one_question_per_image(
            iter_question_array(archive, ANNOTATION_MEMBERS["train"]),
            split="train",
            image_limit=args.train_images,
            seed=args.seed,
        )
        validation = select_one_question_per_image(
            iter_question_array(archive, ANNOTATION_MEMBERS["validation"]),
            split="validation",
            image_limit=args.validation_images,
            seed=args.seed,
            excluded_normalized_questions={
                normalize_question(str(row["question"])) for row in train
            },
        )
    require_group_disjoint(train, validation)
    if len(train) != args.train_images or len(validation) != args.validation_images:
        raise ValueError("source contains fewer unique images than requested")
    train_question_content = {
        normalize_question(str(row["question"])) for row in train
    }
    validation_question_content = {
        normalize_question(str(row["question"])) for row in validation
    }
    content_overlap = train_question_content & validation_question_content
    if content_overlap:
        raise ValueError("normalized question content overlaps across splits")
    args.output.mkdir(parents=True)
    hashes = {}
    for split, rows in (("train", train), ("validation", validation)):
        path = args.output / f"{split}-examples.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        hashes[path.name] = sha256(path)
    metadata = {
        "schema_version": 1,
        "source": "CLEVR v1.0 official train and validation question annotations",
        "source_archive_sha256": archive_sha,
        "seed": args.seed,
        "selection": (
            "one hash-ranked question per hash-ranked image, separately within official split"
        ),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_validation_image_overlap": 0,
        "train_validation_normalized_question_overlap": len(content_overlap),
        "answer_kind_counts": {
            split: dict(Counter(str(row["answer_kind"]) for row in rows))
            for split, rows in (("train", train), ("validation", validation))
        },
        "question_type_counts": {
            split: dict(Counter(str(row["outer_program"]) for row in rows))
            for split, rows in (("train", train), ("validation", validation))
        },
        "option_count_counts": {
            split: dict(Counter(str(len(row["options"])) for row in rows))
            for split, rows in (("train", train), ("validation", validation))
        },
        "manifest_sha256": hashes,
        "media_extracted": False,
        "CoGenT_B_accessed": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
