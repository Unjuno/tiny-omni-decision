import numpy as np
import pytest

from tiny_omni_decision.fusion_diagnostics import complementarity


def test_complementarity_counts_rescue_and_harm():
    a = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    b = [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]]
    j = [[0.1, 0.9], [0.2, 0.8], [0.6, 0.4]]
    result = complementarity(a, b, j, np.array([0, 1, 1]))
    assert result["both_correct"] == 1
    assert result["second_only_correct"] == 1
    assert result["neither_correct"] == 1
    assert result["joint_loses_both_correct"] == 1
    assert result["joint_rescues_first_wrong"] == 1
    assert result["selection_oracle_accuracy"] == 2 / 3
    assert result["label_aware_selection_oracle_nll"] <= result["first_nll"]


@pytest.mark.parametrize("bad", [[[0.8, 0.8]], [[float("nan"), 0.1]], [[-0.1, 1.1]]])
def test_invalid_probabilities_fail(bad):
    with pytest.raises(ValueError):
        complementarity(bad, [[0.5, 0.5]], [[0.5, 0.5]], np.array([0]))
