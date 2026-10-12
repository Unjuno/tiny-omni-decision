import json

import pytest
import torch

from tiny_omni_decision.decision import DecisionOutput, FrozenFeatureCandidateScorer


def result(values):
    p = torch.tensor(values)
    return DecisionOutput(torch.zeros_like(p), p, 0, .5)


def test_named_json_preserves_binary_and_custom_values():
    for labels, values in [(["yes", "no"], [.97, .03]),
                           (["続行", "減速", "停止"], [.05, .8, .15])]:
        output = result(values)
        before = output.probabilities.clone()
        decoded = json.loads(output.to_json('判断"\\', labels))
        assert list(decoded['判断"\\']) == labels
        assert list(decoded['判断"\\'].values()) == before.tolist()
        assert torch.equal(output.probabilities, before)


@pytest.mark.parametrize("values,labels", [
    ([.5, .5], ["same", "same"]), ([.5, .5], ["only"]),
    ([.5, .5], ["", "b"]), ([float("nan"), .5], ["a", "b"]),
    ([.1, .2], ["a", "b"]), ([-.1, 1.1], ["a", "b"]),
])
def test_json_rejects_ambiguous_or_invalid_output(values, labels):
    with pytest.raises(ValueError):
        result(values).to_json("question", labels)


def test_real_scorer_json_needs_no_new_parameters():
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(4, 8).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        output = model.decide(torch.randn(3, 4))
    decoded = json.loads(output.to_json("action", ["a", "b", "c"]))
    assert list(decoded["action"].values()) == output.probabilities.tolist()
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
