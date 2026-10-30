import json

import pytest

from tiny_omni_decision.qasc import (
    check_qasc_split_integrity,
    normalize_qasc_row,
    prepare_qasc_split_rows,
)
from tiny_omni_decision.schema import DatasetManifest


def _row(**overrides):
    row = {
        "id": "qasc-row-001",
        "question": "What does condensation form?",
        "choices": {
            "label": list("BACHEFGD"),
            "text": [
                "water beads",
                "metal",
                "salt",
                "ice",
                "clouds",
                "mist",
                "rain",
                "snow",
            ],
        },
        "answerKey": "B",
        "fact1": "Water vapor can condense.",
        "fact2": "Condensed water forms droplets.",
        "combinedfact": "Water vapor can condense into droplets.",
    }
    row.update(overrides)
    return row


def test_qasc_row_maps_answer_label_to_choice_text_and_keeps_text_only_inputs():
    example = normalize_qasc_row(
        _row(), revision="a" * 40, split="train"
    )

    assert example["options"] == [
        "water beads",
        "metal",
        "salt",
        "ice",
        "clouds",
        "mist",
        "rain",
        "snow",
    ]
    assert example["target"] == "water beads"
    assert example["state"] == ""
    assert example["question"] == "What does condensation form?"
    assert "Water vapor" not in example["state"] + example["question"]
    assert "combinedfact" not in example
    assert example["source_record_id"].endswith(":qasc-row-001")
    assert example["license"] == "CC-BY-4.0"


def test_qasc_fact_pair_group_identity_is_order_independent():
    first = normalize_qasc_row(_row(), revision="a" * 40, split="train")
    reversed_pair = normalize_qasc_row(
        _row(
            fact1="Condensed water forms droplets.",
            fact2="Water vapor can condense.",
        ),
        revision="a" * 40,
        split="train",
    )

    assert first["source_record_id"].split(":", 1)[0] == reversed_pair[
        "source_record_id"
    ].split(":", 1)[0]


def test_qasc_adapter_preserves_near_duplicate_choice_text_without_collapsing_it():
    choices = _row()["choices"]
    choices["text"][1] = "Water beads."
    example = normalize_qasc_row(
        _row(choices=choices), revision="a" * 40, split="train"
    )

    assert example["options"][:2] == ["water beads", "Water beads."]
    assert example["target"] == "water beads"


def test_qasc_split_integrity_counts_disjoint_fact_pair_groups():
    train = normalize_qasc_row(_row(), revision="a" * 40, split="train")
    validation = normalize_qasc_row(
        _row(
            id="qasc-row-002",
            question="What is a second science question?",
            fact1="A different source fact.",
            fact2="Another distinct source fact.",
        ),
        revision="a" * 40,
        split="validation",
    )

    assert check_qasc_split_integrity([train], [validation]) == {
        "train_rows": 1,
        "validation_rows": 1,
        "train_fact_pair_groups": 1,
        "validation_fact_pair_groups": 1,
        "train_validation_group_overlap": 0,
        "train_validation_id_overlap": 0,
        "train_validation_question_overlap": 0,
        "train_validation_content_overlap": 0,
    }


def test_qasc_split_integrity_rejects_shared_fact_pair_group():
    train = normalize_qasc_row(_row(), revision="a" * 40, split="train")
    validation = normalize_qasc_row(
        _row(id="qasc-row-002", question="A new wording from the same facts?"),
        revision="a" * 40,
        split="validation",
    )

    with pytest.raises(ValueError, match="fact-pair groups"):
        check_qasc_split_integrity([train], [validation])


def test_prepare_qasc_split_rows_keeps_only_train_and_validation_inputs():
    train, validation, audit = prepare_qasc_split_rows(
        [_row()],
        [
            _row(
                id="qasc-row-002",
                question="What changes liquid water into vapor?",
                fact1="Heating can evaporate liquid water.",
                fact2="Vapor is gaseous water.",
            )
        ],
        revision="a" * 40,
    )

    assert train[0]["split"] == "train"
    assert validation[0]["split"] == "validation"
    assert audit["train_rows"] == 1
    assert audit["validation_rows"] == 1
    assert audit["train_validation_group_overlap"] == 0


def test_frozen_text_runner_loads_generic_rows_under_split_specific_manifests(tmp_path):
    from tiny_omni_decision.text_probe_data import (
        load_generic_text_examples,
        text_probe_identity_sets,
    )

    revision = "a" * 40
    train_manifest = DatasetManifest(
        dataset_id="allenai/qasc",
        revision=revision,
        split="train",
        modalities=["text"],
        upstream_url="https://huggingface.co/datasets/allenai/qasc",
        license="CC-BY-4.0",
        usage="training",
        trust_status="trusted",
    )
    validation_manifest = train_manifest.model_copy(
        update={"split": "validation", "usage": "evaluation"}
    )
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    same_fact_pair = normalize_qasc_row(
        _row(id="qasc-row-003", question="A reworded question from the same facts?"),
        revision=revision,
        split="train",
    )
    train_path.write_text(
        "\n".join(
            [
                json.dumps(normalize_qasc_row(_row(), revision=revision, split="train")),
                json.dumps(same_fact_pair),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            normalize_qasc_row(
                _row(
                    id="qasc-row-002",
                    question="A different science question?",
                    fact1="A different fact.",
                    fact2="A second different fact.",
                ),
                revision=revision,
                split="validation",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    train = load_generic_text_examples(train_path, train_manifest)
    validation = load_generic_text_examples(validation_path, validation_manifest)

    assert train[0].target == "water beads"
    assert train[0].id.endswith("qasc-row-001")
    assert train[0].source_record_id.split(":", 1)[0] == normalize_qasc_row(
        _row(), revision=revision, split="train"
    )["source_record_id"].split(":", 1)[0]
    assert train[0].source_record_id.split(":", 1)[0] != validation[0].source_record_id.split(
        ":", 1
    )[0]
    groups, _ = text_probe_identity_sets(train)
    assert len(train) == 2
    assert len(groups) == 1


def test_generic_probe_manifest_gate_requires_matching_train_and_eval_usage():
    from tiny_omni_decision.text_probe_data import validate_generic_probe_manifests

    revision = "a" * 40
    training = DatasetManifest(
        dataset_id="allenai/qasc",
        revision=revision,
        split="train",
        modalities=["text"],
        upstream_url="https://huggingface.co/datasets/allenai/qasc",
        license="CC-BY-4.0",
        usage="training",
    )
    evaluation = training.model_copy(update={"split": "validation", "usage": "evaluation"})

    validate_generic_probe_manifests(training, evaluation)
    with pytest.raises(ValueError, match="matching"):
        validate_generic_probe_manifests(
            training, evaluation.model_copy(update={"revision": "b" * 40})
        )
    with pytest.raises(ValueError, match="training-split"):
        validate_generic_probe_manifests(
            training.model_copy(update={"usage": "both"}), evaluation
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"answerKey": "Z"},
        {"choices": {"label": ["A", "A"], "text": ["x", "y"]}},
        {
            "choices": {
                "label": list("ABCDEFGI"),
                "text": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
            }
        },
        {"choices": {"label": ["A", "B"], "text": ["x"]}},
        {
            "choices": {
                "label": list("ABCDEFGH"),
                "text": ["same", "same", "three", "four", "five", "six", "seven", "eight"],
            }
        },
        {"fact1": "", "fact2": ""},
    ],
)
def test_qasc_adapter_rejects_malformed_target_choices_or_group_source(overrides):
    with pytest.raises(ValueError):
        normalize_qasc_row(_row(**overrides), revision="a" * 40, split="train")
