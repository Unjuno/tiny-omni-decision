import numpy as np
import pytest

from tiny_omni_decision.robomnist_av_match_evaluation import (
    classification_metrics,
    paired_bootstrap_delta,
    validate_examples,
)


def test_validate_examples_requires_grouped_balanced_pairs_and_no_split_overlap():
    pair = "p1"
    train = [
        {
            "pair_unit_id": pair,
            "condition": "c",
            "audio_group": "a",
            "video_group": "a",
            "target": 1,
        },
        {
            "pair_unit_id": pair,
            "condition": "c",
            "audio_group": "a",
            "video_group": "b",
            "target": 0,
        },
        {
            "pair_unit_id": pair,
            "condition": "c",
            "audio_group": "b",
            "video_group": "b",
            "target": 1,
        },
        {
            "pair_unit_id": pair,
            "condition": "c",
            "audio_group": "b",
            "video_group": "a",
            "target": 0,
        },
    ]
    validation = [
        {
            "pair_unit_id": "p2",
            "condition": "c",
            "audio_group": "x",
            "video_group": "x",
            "target": 1,
        },
        {
            "pair_unit_id": "p2",
            "condition": "c",
            "audio_group": "x",
            "video_group": "y",
            "target": 0,
        },
        {
            "pair_unit_id": "p2",
            "condition": "c",
            "audio_group": "y",
            "video_group": "y",
            "target": 1,
        },
        {
            "pair_unit_id": "p2",
            "condition": "c",
            "audio_group": "y",
            "video_group": "x",
            "target": 0,
        },
    ]

    result = validate_examples({"examples": {"train": train, "validation": validation}})

    assert len(result["train"]) == len(result["validation"]) == 4


def test_validate_examples_rejects_target_that_disagrees_with_group_identity():
    row = {
        "pair_unit_id": "p",
        "condition": "c",
        "audio_group": "a",
        "video_group": "a",
        "target": 0,
    }
    with pytest.raises(ValueError, match="target disagrees"):
        validate_examples({"examples": {"train": [row], "validation": []}})


def test_metrics_report_accuracy_calibration_and_confidence():
    probabilities = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    result = classification_metrics(probabilities, np.asarray([1, 0]))

    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["nll"] == pytest.approx(-np.log([0.9, 0.8]).mean())
    assert result["brier"] == pytest.approx(0.05)
    assert result["mean_confidence"] == pytest.approx(0.85)


def test_paired_bootstrap_requires_independent_pair_unit_floor():
    with pytest.raises(ValueError, match="at least 200 independent"):
        paired_bootstrap_delta(
            np.asarray([True, False]),
            np.asarray([False, True]),
            np.asarray(["p1", "p1"]),
            replicates=10,
        )
