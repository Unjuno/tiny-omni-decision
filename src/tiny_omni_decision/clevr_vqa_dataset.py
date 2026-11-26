"""Deterministic image-grouped examples for the official CLEVR VQA task."""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from collections.abc import Iterable, Iterator
from typing import Any

from tiny_omni_decision.clevr_vqa_options import answer_kind, make_options

ANNOTATION_MEMBERS = {
    "train": "CLEVR_v1.0/questions/CLEVR_train_questions.json",
    "validation": "CLEVR_v1.0/questions/CLEVR_val_questions.json",
}


def iter_question_array(archive: zipfile.ZipFile, member: str) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON ``questions`` array in a ZIP member."""
    decoder = json.JSONDecoder()
    stream = io.TextIOWrapper(archive.open(member), encoding="utf-8")
    buffer = ""
    position = 0
    started = False
    while True:
        if not started:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError(f"missing questions array in {member}")
            buffer += chunk
            marker = buffer.find('"questions"')
            if marker < 0:
                continue
            start = buffer.find("[", marker)
            while start < 0:
                more = stream.read(65536)
                if not more:
                    raise ValueError(f"missing questions array in {member}")
                buffer += more
                start = buffer.find("[", marker)
            position = start + 1
            started = True
        while True:
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                row, end = decoder.raw_decode(buffer, position)
                position = end
                if not isinstance(row, dict):
                    raise ValueError("question array contains a non-object row")
                yield row
                break
            except json.JSONDecodeError as exc:
                more = stream.read(65536)
                if not more:
                    if not buffer[position:].strip():
                        return
                    raise ValueError(f"truncated questions array in {member}") from exc
                buffer = buffer[position:] + more
                position = 0


def _hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def normalize_question(question: str) -> str:
    """Normalize generated CLEVR question text for the cross-split content gate."""
    return " ".join(question.casefold().split())


def grouped_shuffle_indices(rows: list[dict[str, Any]], seed: int = 17) -> list[int]:
    """Return a deterministic within-split derangement of image row indices."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        split = row.get("split")
        if not isinstance(split, str):
            raise ValueError("every row must identify its split")
        groups.setdefault(split, []).append(index)
    mapping = list(range(len(rows)))
    for split, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f"split {split!r} needs at least two images for shuffling")
        rng = random.Random(f"clevr-image-shuffle-v1|{seed}|{split}")
        permutation = list(indices)
        for position in range(len(permutation) - 1, 0, -1):
            other = rng.randrange(position)
            permutation[position], permutation[other] = permutation[other], permutation[position]
        if any(
            original == shuffled
            for original, shuffled in zip(indices, permutation, strict=True)
        ):
            raise RuntimeError("Sattolo shuffle unexpectedly produced a fixed point")
        for original, shuffled in zip(indices, permutation, strict=True):
            mapping[original] = shuffled
    return mapping


def select_one_question_per_image(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    image_limit: int,
    seed: int = 17,
    excluded_normalized_questions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select one hash-ranked source question per image, preserving image grouping."""
    if split not in ANNOTATION_MEMBERS:
        raise ValueError("split must be train or validation")
    if image_limit <= 0:
        raise ValueError("image_limit must be positive")
    best_by_image: dict[str, tuple[int, dict[str, Any]]] = {}
    for row in rows:
        filename = row.get("image_filename")
        question_index = row.get("question_index")
        answer = row.get("answer")
        question = row.get("question")
        if not isinstance(filename, str) or not isinstance(question_index, int):
            raise ValueError("source row lacks image identity or question index")
        if not isinstance(answer, str) or not isinstance(question, str):
            raise ValueError("source row lacks answer or question")
        if excluded_normalized_questions and normalize_question(question) in (
            excluded_normalized_questions
        ):
            continue
        score = _hash_int(f"CLEVR-VQA-question|{seed}|{filename}|{question_index}")
        if filename not in best_by_image or score < best_by_image[filename][0]:
            best_by_image[filename] = (score, row)
    selected_images = sorted(
        best_by_image,
        key=lambda filename: (_hash_int(f"CLEVR-VQA-image|{seed}|{split}|{filename}"), filename),
    )[:image_limit]
    result = []
    for filename in selected_images:
        source = best_by_image[filename][1]
        options = make_options(
            image_filename=filename,
            question_index=source["question_index"],
            answer=source["answer"],
            seed=seed,
        )
        result.append(
            {
                "sample_id": f"{filename}:{source['question_index']}",
                "split": split,
                "image_filename": filename,
                "question_index": source["question_index"],
                "question": source["question"],
                "answer": source["answer"],
                "answer_kind": answer_kind(source["answer"]),
                "outer_program": (
                    source["program"][-1].get("function", "unknown")
                    if isinstance(source.get("program"), list)
                    and source["program"]
                    and isinstance(source["program"][-1], dict)
                    else "unknown"
                ),
                "options": list(options),
                "target_position": options.index(source["answer"]),
            }
        )
    return result


def select_cogent_b_audit_examples(
    rows: Iterable[dict[str, Any]],
    *,
    image_limit: int,
    seed: int = 17,
    excluded_normalized_questions: set[str],
    forbidden_image_filenames: set[str],
) -> list[dict[str, Any]]:
    """Select one hash-ranked, content-disjoint question for every CoGenT-B image.

    The function fails closed if any expected image lacks a question whose
    normalized text is absent from development data. It never shortens the
    audit to a subset of easier or more convenient images.
    """
    if image_limit <= 0:
        raise ValueError("image_limit must be positive")
    best_by_image: dict[str, tuple[int, dict[str, Any]]] = {}
    seen_source_ids: set[tuple[str, int]] = set()
    source_images: set[str] = set()
    for row in rows:
        filename = row.get("image_filename")
        question_index = row.get("question_index")
        answer = row.get("answer")
        question = row.get("question")
        if not isinstance(filename, str) or not filename.startswith("CLEVR_valB_"):
            raise ValueError("CoGenT-B audit rows must use valB image identities")
        if not isinstance(question_index, int) or not isinstance(answer, str):
            raise ValueError("CoGenT-B question row lacks a valid index or answer")
        if not isinstance(question, str):
            raise ValueError("CoGenT-B question row lacks question text")
        source_id = (filename, question_index)
        if source_id in seen_source_ids:
            raise ValueError("CoGenT-B source question identity is duplicated")
        seen_source_ids.add(source_id)
        source_images.add(filename)
        if filename in forbidden_image_filenames:
            raise ValueError("CoGenT-B audit image overlaps a development image")
        if normalize_question(question) in excluded_normalized_questions:
            continue
        score = _hash_int(
            f"CLEVR-CoGenT-B-audit-question-v1|{seed}|{filename}|{question_index}"
        )
        if filename not in best_by_image or score < best_by_image[filename][0]:
            best_by_image[filename] = (score, row)

    if len(source_images) != image_limit or len(best_by_image) != image_limit:
        raise ValueError(
            "CoGenT-B audit requires one content-disjoint question for every source image; "
            f"found {len(source_images)} source images and {len(best_by_image)} eligible images, "
            f"expected {image_limit}"
        )

    selected_images = sorted(
        best_by_image,
        key=lambda filename: (
            _hash_int(f"CLEVR-CoGenT-B-audit-image-v1|{seed}|{filename}"),
            filename,
        ),
    )
    result = []
    for filename in selected_images:
        source = best_by_image[filename][1]
        answer = source["answer"]
        options = make_options(
            image_filename=filename,
            question_index=source["question_index"],
            answer=answer,
            seed=seed,
        )
        program = source.get("program")
        outer_program = (
            program[-1].get("function", "unknown")
            if isinstance(program, list) and program and isinstance(program[-1], dict)
            else "unknown"
        )
        result.append(
            {
                "sample_id": f"{filename}:{source['question_index']}",
                "split": "cogent_b_audit",
                "image_filename": filename,
                "question_index": source["question_index"],
                "question": source["question"],
                "answer": answer,
                "answer_kind": answer_kind(answer),
                "outer_program": outer_program,
                "options": list(options),
                "target_position": options.index(answer),
            }
        )
    if len({row["sample_id"] for row in result}) != image_limit:
        raise ValueError("CoGenT-B audit selection repeated a source question")
    if len({row["image_filename"] for row in result}) != image_limit:
        raise ValueError("CoGenT-B audit selection repeated an image")
    if {
        normalize_question(str(row["question"])) for row in result
    } & excluded_normalized_questions:
        raise ValueError("CoGenT-B audit selection overlaps normalized development questions")
    return result


def require_group_disjoint(
    train: Iterable[dict[str, Any]], validation: Iterable[dict[str, Any]]
) -> None:
    """Reject sample or underlying-image overlap between train and validation."""
    train_rows = list(train)
    validation_rows = list(validation)
    for field in ("sample_id", "image_filename"):
        train_values = {str(row[field]) for row in train_rows}
        validation_values = {str(row[field]) for row in validation_rows}
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(f"train/validation {field} overlap: {sorted(overlap)[:3]}")
