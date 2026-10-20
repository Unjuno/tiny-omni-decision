import numpy as np
import pytest

from scripts.report_pch_lstm_bootstrap import paired_subject_bootstrap


def test_paired_subject_bootstrap_names_models_and_delta_direction():
    first = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    second = np.asarray([[0.6, 0.4], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]])
    targets = np.asarray([0, 0, 1, 1])
    subjects = ["p1", "p1", "p2", "p2"]

    result = paired_subject_bootstrap(
        first,
        second,
        targets,
        subjects,
        name_a="LSTM",
        name_b="MOMENT",
        seed=17,
        replicates=200,
    )

    assert result["delta_direction"] == "LSTM minus MOMENT"
    assert result["delta_intervals"]["accuracy"] == [0.0, 0.5]
    assert result["participants"] == 2


def test_paired_subject_bootstrap_rejects_invalid_probability_rows():
    probabilities = np.asarray([[0.4, 0.4], [0.2, 0.8]])

    with pytest.raises(ValueError, match="sum to one"):
        paired_subject_bootstrap(
            probabilities,
            probabilities,
            np.asarray([0, 1]),
            ["p1", "p2"],
            name_a="A",
            name_b="B",
        )
