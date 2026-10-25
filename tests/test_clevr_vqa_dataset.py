import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _rows(split, count=6, question_variants=3):
    return [
        {
            "image_filename": f"CLEVR_{split}_{image:06}.png",
            "question_index": question,
            "question": f"How many objects? variant {question}",
            "answer": str((image + question) % 3),
            "program": [{"function": "count"}],
        }
        for image in range(count)
        for question in range(question_variants)
    ]


def test_streams_json_questions_array_from_zip(tmp_path):
    from tiny_omni_decision.clevr_vqa_dataset import iter_question_array

    archive_path = tmp_path / "annotations.zip"
    rows = _rows("train", 4)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "questions.json",
            json.dumps({"info": {"version": "test"}, "questions": rows}),
        )
    with zipfile.ZipFile(archive_path) as archive:
        assert list(iter_question_array(archive, "questions.json")) == rows


def test_annotation_dataset_import_does_not_require_torch():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('torch intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tiny_omni_decision.clevr_vqa_dataset import select_one_question_per_image
assert callable(select_one_question_per_image)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_selection_is_deterministic_and_groups_by_image():
    from tiny_omni_decision.clevr_vqa_dataset import select_one_question_per_image

    source = _rows("train")
    selected = select_one_question_per_image(source, split="train", image_limit=4, seed=17)
    replay = select_one_question_per_image(source, split="train", image_limit=4, seed=17)
    assert selected == replay
    assert len(selected) == len({row["image_filename"] for row in selected}) == 4
    assert all(row["answer"] in row["options"] for row in selected)
    assert all(len(row["options"]) in (2, 3, 4) for row in selected)


def test_train_validation_media_overlap_is_rejected():
    from tiny_omni_decision.clevr_vqa_dataset import require_group_disjoint

    with pytest.raises(ValueError, match="image_filename overlap"):
        require_group_disjoint(
            [{"sample_id": "train:1", "image_filename": "shared.png"}],
            [{"sample_id": "val:2", "image_filename": "shared.png"}],
        )


def test_validation_selection_can_exclude_normalized_train_question_content():
    from tiny_omni_decision.clevr_vqa_dataset import (
        normalize_question,
        select_one_question_per_image,
    )

    train = select_one_question_per_image(
        _rows("train", count=8, question_variants=8),
        split="train",
        image_limit=4,
        seed=17,
    )
    train_content = {normalize_question(row["question"]) for row in train}
    validation = select_one_question_per_image(
        _rows("validation", count=8, question_variants=8),
        split="validation",
        image_limit=4,
        seed=17,
        excluded_normalized_questions=train_content,
    )
    assert len(validation) == 4
    assert not train_content & {normalize_question(row["question"]) for row in validation}


def test_cogent_b_audit_selection_is_deterministic_and_content_disjoint():
    from tiny_omni_decision.clevr_vqa_dataset import (
        normalize_question,
        select_cogent_b_audit_examples,
    )

    rows = [
        {
            "image_filename": f"CLEVR_valB_{image:06}.png",
            "question_index": question,
            "question": text,
            "answer": str((image + question) % 3),
            "program": [{"function": "count"}],
        }
        for image in range(4)
        for question, text in enumerate(("Seen in development", f"novel {image}"))
    ]
    excluded = {normalize_question("seen in development")}

    selected = select_cogent_b_audit_examples(
        rows,
        image_limit=4,
        seed=17,
        excluded_normalized_questions=excluded,
        forbidden_image_filenames=set(),
    )

    assert selected == select_cogent_b_audit_examples(
        rows,
        image_limit=4,
        seed=17,
        excluded_normalized_questions=excluded,
        forbidden_image_filenames=set(),
    )
    assert len({row["sample_id"] for row in selected}) == 4
    assert len({row["image_filename"] for row in selected}) == 4
    assert all(row["split"] == "cogent_b_audit" for row in selected)
    assert all(row["answer"] in row["options"] for row in selected)
    assert not excluded & {normalize_question(row["question"]) for row in selected}


def test_cogent_b_audit_selection_fails_closed_without_full_image_coverage():
    from tiny_omni_decision.clevr_vqa_dataset import select_cogent_b_audit_examples

    rows = [
        {
            "image_filename": "CLEVR_valB_000000.png",
            "question_index": 0,
            "question": "Seen in development",
            "answer": "yes",
        },
        {
            "image_filename": "CLEVR_valB_000001.png",
            "question_index": 0,
            "question": "novel question",
            "answer": "no",
        },
    ]
    with pytest.raises(ValueError, match="every source image"):
        select_cogent_b_audit_examples(
            rows,
            image_limit=2,
            seed=17,
            excluded_normalized_questions={"seen in development"},
            forbidden_image_filenames=set(),
        )


def test_cogent_b_audit_selection_rejects_development_asset_overlap():
    from tiny_omni_decision.clevr_vqa_dataset import select_cogent_b_audit_examples

    row = {
        "image_filename": "CLEVR_valB_000000.png",
        "question_index": 0,
        "question": "novel question",
        "answer": "yes",
    }
    with pytest.raises(ValueError, match="overlaps a development image"):
        select_cogent_b_audit_examples(
            [row],
            image_limit=1,
            seed=17,
            excluded_normalized_questions=set(),
            forbidden_image_filenames={row["image_filename"]},
        )


def test_grouped_shuffle_is_deterministic_deranged_and_split_local():
    from tiny_omni_decision.clevr_vqa_dataset import grouped_shuffle_indices

    rows = [
        {"split": split, "image_filename": f"{split}-{index}.png"}
        for split in ("train", "validation")
        for index in range(8)
    ]
    mapping = grouped_shuffle_indices(rows, seed=17)
    assert mapping == grouped_shuffle_indices(rows, seed=17)
    assert all(index != other for index, other in enumerate(mapping))
    assert all(rows[index]["split"] == rows[other]["split"] for index, other in enumerate(mapping))
    with pytest.raises(ValueError, match="at least two"):
        grouped_shuffle_indices([{"split": "validation"}], seed=17)
