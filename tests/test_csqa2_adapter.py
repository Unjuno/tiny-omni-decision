import gzip
import json

from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl
from tiny_omni_decision.schema import DatasetManifest

REVISION = "3f57a4a6a1d51fe6a9b462b59be67c08d41fdc39"


def manifest(split="train"):
    return DatasetManifest.model_validate(
        {
            "dataset_id": "allenai/csqa2",
            "revision": REVISION,
            "split": split,
            "modalities": ["text"],
            "upstream_url": "https://github.com/allenai/csqa2",
            "license": "CC-BY-4.0",
            "commercial_use": True,
            "derivative_model_training_allowed": True,
            "redistribution_allowed": True,
            "usage": "training" if split == "train" else "evaluation",
            "trust_status": "trusted",
            "attribution": "Allen Institute for AI, CommonsenseQA 2.0, CC BY 4.0",
        }
    )


def row(record_id="r1", question="Can a robin fly?", answer="yes", split="train"):
    return {
        "id": record_id,
        "question": question,
        "answer": answer,
        "confidence": 0.89,
        "date": "2021-01-01",
        "relational_prompt": "",
        "topic_prompt": "",
        "relational_prompt_used": False,
        "topic_prompt_used": False,
        "validations": ["yes", "yes"],
        "split": split,
    }


def test_csqa2_maps_yes_no_to_stable_caller_options():
    examples = list(normalize_jsonl([row()], manifest(), "csqa2"))

    assert len(examples) == 1
    example = examples[0]
    assert example.options == ["no", "yes"]
    assert example.target == "yes"
    assert example.state == ""
    assert example.question == "Can a robin fly?"
    assert example.id == "allenai/csqa2:r1"
    assert example.source_record_id == "r1"
    assert example.source_target == "yes"
    assert example.modality == "text"


def test_csqa2_rejects_nonbinary_source_labels():
    try:
        list(normalize_jsonl([row(answer="maybe")], manifest(), "csqa2"))
    except ValueError as exc:
        assert "yes or no" in str(exc)
    else:
        raise AssertionError("a non-binary source label must not be coerced")


def test_csqa2_keeps_same_split_duplicate_questions_as_separate_examples():
    examples = list(
        normalize_jsonl(
            [row("r1", "A robin is a bird."), row("r2", "A robin is a bird.", "no")],
            manifest(),
            "csqa2",
        )
    )

    assert [example.source_record_id for example in examples] == ["r1", "r2"]
    assert [example.target for example in examples] == ["yes", "no"]


def test_csqa2_does_not_admit_rows_marked_for_another_split():
    examples = list(normalize_jsonl([row(split="validation")], manifest("train"), "csqa2"))

    assert examples == []


def test_iter_local_rows_reads_gzip_jsonl_in_source_order(tmp_path):
    expected = [row("first"), row("second", answer="no")]
    for filename in ("csqa2.jsonl.gz", "CSQA2_train.json.gz"):
        path = tmp_path / filename
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for item in expected:
                handle.write(json.dumps(item) + "\n")
        assert list(iter_local_rows(path)) == expected
