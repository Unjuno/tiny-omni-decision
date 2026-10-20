from __future__ import annotations

import importlib

import pytest

_audit = importlib.import_module("scripts.audit_clevrer_temporal_readout")


def _row(sample_id: str, split: str, scene: str, *, question: str = "Which came first?") -> dict:
    return {
        "id": sample_id,
        "split": split,
        "scene_group_id": scene,
        "media_sha256": f"sha-{scene}",
        "question": question,
        "options": ["no", "yes"],
        "target": "yes",
        "taxonomy": "exist",
        "question_type": "temporal_descriptive",
        "program": ["events", "filter_before", "exist"],
    }


def _prediction(row: dict) -> dict:
    return {
        "id": row["id"],
        "target": row["target"],
        "prediction": "yes",
        "options": [
            {"option_id": "no", "probability": 0.2},
            {"option_id": "yes", "probability": 0.8},
        ],
    }


def test_audit_accepts_disjoint_data_and_exact_saved_predictions() -> None:
    train = [_row("tr-1", "train", "scene-tr")]
    validation = [_row("va-1", "validation", "scene-va", question="Did any object enter?")]

    result = _audit.audit(train, validation, [_prediction(validation[0])])

    assert result["counts"]["scene_overlap"] == 0
    assert result["saved_readout_validation"]["accuracy"] == 1.0
    assert result["saved_readout_by_operator_family"]["order"]["order"]["n"] == 1


@pytest.mark.parametrize(
    ("train_mutation", "error"),
    [
        (
            lambda train, validation: train.__setitem__(0, _row("tr-1", "train", "scene-va")),
            "scene",
        ),
        (
            lambda train, validation: train.__setitem__(
                0, _row("tr-1", "train", "scene-tr", question=validation[0]["question"])
            ),
            "normalized",
        ),
    ],
)
def test_audit_rejects_cross_split_group_or_content_overlap(train_mutation, error: str) -> None:
    train = [_row("tr-1", "train", "scene-tr")]
    validation = [_row("va-1", "validation", "scene-va", question="Did any object enter?")]
    train_mutation(train, validation)

    with pytest.raises(ValueError, match=error):
        _audit.audit(train, validation, [_prediction(validation[0])])


def test_audit_rejects_prediction_id_or_probability_corruption() -> None:
    train = [_row("tr-1", "train", "scene-tr")]
    validation = [_row("va-1", "validation", "scene-va", question="Did any object enter?")]
    with pytest.raises(ValueError, match="exactly match"):
        _audit.audit(train, validation, [])

    bad = _prediction(validation[0])
    bad["options"][0]["probability"] = 0.4
    with pytest.raises(ValueError, match="sum to one"):
        _audit.audit(train, validation, [bad])
