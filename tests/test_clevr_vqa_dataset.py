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
