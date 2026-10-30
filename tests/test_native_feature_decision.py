from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.native_feature_decision import (  # noqa: E402
    EnumOptionField,
    NativeFeatureDecisionScorer,
    NumericOptionField,
    TypedOptionSchema,
    decide_cached_native_feature,
)
from tiny_omni_decision.observations import NativeOption, ObservationBatch  # noqa: E402


def cached_request(tmp_path, *, valid=(True, True), criterion=None):
    path = tmp_path / "features.npz"
    values = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=np.float32)
    timestamps = np.array([10., 20.])
    mask = np.array(valid, dtype=np.bool_)
    np.savez(path, values=values, timestamps_ms=timestamps, valid_mask=mask)
    batch = ObservationBatch.model_validate({
        "schema_version": "1.0",
        "observations": [{
            "modality": "video", "source_id": "camera",
            "payload_ref": str(path),
            "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clock_id": "clock-a", "timestamps_ms": (10., 20.),
            "valid_mask": valid,
        }],
        "options": [
            {"option_id": "caller-a", "attributes": {"action_code": 1}},
            {"option_id": "caller-b", "attributes": {"action_code": 2}},
        ],
        "criterion": criterion,
    })
    return batch, path


def test_text_free_native_scorer_is_candidate_permutation_equivariant():
    torch.manual_seed(17)
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=2, hidden_width=8).eval()
    state = torch.tensor([[0.2, -0.3, 0.7]])
    candidates = torch.tensor([[[1., 0.], [0., 1.], [2., -1.]]])
    valid = torch.tensor([[True, True, True]])

    logits = model(state, candidates, valid)
    order = torch.tensor([2, 0, 1])
    shuffled = model(state, candidates[:, order], valid[:, order])
    torch.testing.assert_close(shuffled, logits[:, order], rtol=0, atol=0)


def test_decide_preserves_supplied_option_order_and_returns_typed_json():
    torch.manual_seed(17)
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=2, hidden_width=8).eval()
    result = model.decide(
        torch.tensor([0.2, -0.3, 0.7]),
        torch.tensor([[1., 0.], [0., 1.], [2., -1.]]),
        ["caller-b", "caller-a", "caller-c"],
        ["action-2", "action-1", "action-3"],
    )
    assert result["status"] == "selected"
    assert [row["option_id"] for row in result["options"]] == [
        "caller-b", "caller-a", "caller-c"
    ]
    assert sum(row["probability"] for row in result["options"]) == pytest.approx(1.0)
    assert result["selected_option_id"] in {"caller-a", "caller-b", "caller-c"}


def test_invalid_padding_is_ignored_but_available_values_must_be_finite():
    model = NativeFeatureDecisionScorer(feature_width=2, option_width=1, hidden_width=4)
    state = torch.tensor([[1., 2.]])
    candidates = torch.tensor([[[0.], [1.], [float("nan")]]])
    valid = torch.tensor([[True, True, False]])
    output = model(state, candidates, valid)
    assert torch.isfinite(output[valid]).all()
    assert torch.isneginf(output[~valid]).all()

    valid[0, 2] = True
    with pytest.raises(ValueError, match="finite"):
        model(state, candidates, valid)


def test_rejects_candidate_set_or_metadata_mismatch():
    model = NativeFeatureDecisionScorer(feature_width=2, option_width=1, hidden_width=4)
    with pytest.raises(ValueError, match="at least two"):
        model(torch.zeros(1, 2), torch.zeros(1, 2, 1), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="unique nonempty"):
        model.decide(torch.zeros(2), torch.zeros(2, 1), ["same", "same"], ["a", "b"])


def test_cached_single_modality_path_checks_payload_and_returns_option_json(tmp_path):
    torch.manual_seed(17)
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=1, hidden_width=8).eval()
    batch, _ = cached_request(tmp_path)
    result = decide_cached_native_feature(
        model, batch,
        modality="video", query_time_ms=20., max_age_ms=15.,
        option_schema=TypedOptionSchema(numeric=(NumericOptionField("action_code"),)),
    )
    assert result["status"] == "selected"
    assert [row["option_id"] for row in result["options"]] == ["caller-a", "caller-b"]


@pytest.mark.parametrize("query_time", [5., 100.])
def test_cached_single_modality_path_abstains_without_fresh_valid_evidence(
    tmp_path, query_time
):
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=1, hidden_width=4).eval()
    batch, _ = cached_request(tmp_path, valid=(False, False))
    result = decide_cached_native_feature(
        model, batch, torch.tensor([[1.], [2.]]),
        modality="video", query_time_ms=query_time, max_age_ms=15.,
    )
    assert result["status"] == "abstain"
    assert result["selected_option_id"] is None
    assert [row["probability"] for row in result["options"]] == [None, None]


def test_cached_single_modality_path_rejects_modified_payload_and_text_condition(tmp_path):
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=1, hidden_width=4).eval()
    batch, path = cached_request(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        decide_cached_native_feature(
            model, batch, torch.tensor([[1.], [2.]]), ["walk", "stop"],
            modality="video", query_time_ms=20., max_age_ms=15.,
        )

    batch, _ = cached_request(
        tmp_path, criterion={"kind": "text", "value": "choose fastest"}
    )
    with pytest.raises(ValueError, match="criterion"):
        decide_cached_native_feature(
            model, batch, torch.tensor([[1.], [2.]]), ["walk", "stop"],
            modality="video", query_time_ms=20., max_age_ms=15.,
        )


def test_typed_option_schema_normalizes_numeric_and_one_hot_encodes_enum():
    schema = TypedOptionSchema(
        numeric=(NumericOptionField("cost", center=10., scale=5.),),
        enums=(EnumOptionField("action", ("wait", "go")),),
    )
    options = [
        NativeOption(option_id="opt-a", attributes={"cost": 5, "action": "wait"}),
        NativeOption(option_id="opt-b", attributes={"cost": 15, "action": "go"}),
    ]
    encoded = schema.encode(options)
    torch.testing.assert_close(encoded, torch.tensor([[-1., 1., 0.], [1., 0., 1.]]))


def test_typed_option_schema_rejects_free_text_and_unknown_categories():
    schema = TypedOptionSchema(enums=(EnumOptionField("action", ("wait", "go")),))
    with pytest.raises(ValueError, match="exactly match"):
        schema.encode([
            NativeOption(option_id="a", attributes={"action": "wait", "label": "secret"}),
            NativeOption(option_id="b", attributes={"action": "go", "label": "secret"}),
        ])
    with pytest.raises(ValueError, match="fixed vocabulary"):
        schema.encode([
            NativeOption(option_id="a", attributes={"action": "stop"}),
            NativeOption(option_id="b", attributes={"action": "go"}),
        ])
