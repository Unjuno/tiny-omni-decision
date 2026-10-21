import json

import pytest

pytest.importorskip("torch")

import torch

from tiny_omni_decision.decision import DecisionOutput, FrozenFeatureCandidateScorer


def result(values):
    p = torch.tensor(values)
    index = int(p.argmax().item())
    return DecisionOutput(torch.zeros_like(p), p, index, float(p[index].item()))


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
    typed = json.loads(output.to_decision_json(
        "request-1", "choice", ["a-id", "b-id", "c-id"], ["a", "b", "c"]
    ))
    assert list(decoded["action"].values()) == output.probabilities.tolist()
    assert typed["selected_option_id"] == ["a-id", "b-id", "c-id"][output.predicted_index]
    assert [item["probability"] for item in typed["options"]] == output.probabilities.tolist()
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_typed_decision_json_keeps_stable_ids_and_candidate_order():
    output = result([.1, .7, .2])
    decoded = json.loads(output.to_decision_json(
        request_id="req-17",
        decision_type="choice",
        option_ids=["continue", "slow", "stop"],
        option_texts=["続行", "減速", "停止"],
    ))

    assert decoded == {
        "schema_version": 1,
        "request_id": "req-17",
        "decision_type": "choice",
        "selected_option_id": "slow",
        "confidence": pytest.approx(.7),
        "options": [
            {"id": "continue", "text": "続行", "probability": pytest.approx(.1)},
            {"id": "slow", "text": "減速", "probability": pytest.approx(.7)},
            {"id": "stop", "text": "停止", "probability": pytest.approx(.2)},
        ],
    }


@pytest.mark.parametrize("decision_type,ids,texts", [
    ("yes_no", ["yes", "no"], ["はい", "いいえ"]),
    ("ordinal_score", ["0", "1", "2"], ["低", "中", "高"]),
])
def test_typed_decision_json_accepts_binary_and_ordinal_shapes(decision_type, ids, texts):
    values = [.2, .8] if len(ids) == 2 else [.2, .3, .5]
    decoded = json.loads(
        result(values).to_decision_json(
            "request", decision_type, ids, texts, ordinal_values=[0, 1, 2]
            if decision_type == "ordinal_score" else None
        )
    )
    assert decoded["decision_type"] == decision_type
    assert len(decoded["options"]) == len(ids)
    if decision_type == "ordinal_score":
        assert decoded["expected_score"] == pytest.approx(1.3)


def test_ordinal_json_requires_ordered_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        result([.2, .3, .5]).to_decision_json(
            "request", "ordinal_score", ["a", "b", "c"], ["A", "B", "C"],
            ordinal_values=[0, 2, 1],
        )


@pytest.mark.parametrize("decision_type,ids,texts", [
    ("unknown", ["yes", "no"], ["yes", "no"]),
    ("yes_no", ["yes"], ["yes"]),
    ("choice", ["same", "same"], ["a", "b"]),
    ("ordinal_score", ["low", "high"], ["low", " "]),
    ("choice", ["a", "b"], ["a"]),
])
def test_typed_decision_json_rejects_invalid_contract(decision_type, ids, texts):
    with pytest.raises(ValueError):
        values = [.5, .5] if len(ids) != 3 else [.2, .3, .5]
        result(values).to_decision_json("request", decision_type, ids, texts)
