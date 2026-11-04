import json

import pytest

from tiny_omni_decision.classification_json import classification_json


def test_json_preserves_semantic_probabilities_under_option_permutation():
    options = [{"option_id": "yes", "class_id": "A01"},
               {"option_id": "no", "class_id": "A02"}]
    result = classification_json([.5, .5], ["A01", "A02"], options)
    reverse = classification_json([.5, .5], ["A01", "A02"], options[::-1])
    assert result["selected_option_id"] == reverse["selected_option_id"] == "yes"
    assert result["options"] == reverse["options"][::-1]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("probabilities", [[float("nan"), .5], [-.1, 1.1], [.1, .1]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        classification_json(probabilities, ["a", "b"], [
            {"option_id": "a", "class_id": "a"}, {"option_id": "b", "class_id": "b"}])


def test_cannot_silently_shrink_fixed_benchmark_options():
    with pytest.raises(ValueError):
        classification_json([.5, .5], ["a", "b"], [{"option_id": "a", "class_id": "a"}])
