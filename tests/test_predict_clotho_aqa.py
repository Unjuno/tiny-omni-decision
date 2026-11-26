import numpy as np
import pytest

from scripts.predict_clotho_aqa import candidate_features, option_order


def test_canonical_no_yes_order():
    assert option_order(["no-id", "yes-id"], ["No", "Yes"]) == [0, 1]
    assert option_order(["yes-id", "no-id"], ["Yes", "No"]) == [1, 0]


@pytest.mark.parametrize(
    ("option_ids", "option_texts"),
    [
        (["same", "same"], ["No", "Yes"]),
        (["a", "b", "c"], ["No", "Yes"]),
        (["a", "b"], ["Yes", "Maybe"]),
    ],
)
def test_rejects_ambiguous_or_unsupported_options(option_ids, option_texts):
    with pytest.raises(ValueError):
        option_order(option_ids, option_texts)


def test_candidate_feature_rows_keep_semantics_and_distinguish_options():
    audio = np.full(384, 2, dtype=np.float32)
    question = np.full(384, 3, dtype=np.float32)

    features = candidate_features(audio, question)

    assert features.shape == (2, 770)
    assert np.all(features[:, :384] == 2)
    assert np.all(features[:, 384:768] == 3)
    assert np.array_equal(features[:, 768:], np.eye(2, dtype=np.float32))


def test_candidate_features_reject_nonfinite_input():
    audio = np.zeros(384, dtype=np.float32)
    question = np.zeros(384, dtype=np.float32)
    question[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        candidate_features(audio, question)
