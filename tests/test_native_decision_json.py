import json

import pytest

from tiny_omni_decision.native_decision_json import native_decision_json
from tiny_omni_decision.observations import ObservationBatch


def request():
    return ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [
                {
                    "modality": "image",
                    "source_id": "frame-0",
                    "payload_ref": "cache:frame-0",
                    "payload_sha256": "a" * 64,
                    "clock_id": "clip-ms",
                    "timestamps_ms": [0.0],
                    "valid_mask": [True],
                },
                {
                    "modality": "sensor",
                    "source_id": "imu-0",
                    "payload_ref": "cache:imu-0",
                    "payload_sha256": "b" * 64,
                    "clock_id": "clip-ms",
                    "timestamps_ms": [0.0, 20.0],
                    "valid_mask": [True, True],
                    "channels": [
                        {"name": "acc_x", "unit": "g", "scale": 1.0, "offset": 0.0}
                    ],
                },
            ],
            "options": [
                {"option_id": "stop", "attributes": {"action": "stop", "safe": True}},
                {"option_id": "go", "attributes": {"action": "go", "safe": False}},
            ],
        }
    )


def test_native_json_keeps_typed_options_and_aligned_probabilities():
    result = native_decision_json(request(), [0.2, 0.8])

    assert result == {
        "schema_version": "1.0",
        "status": "selected",
        "selected_option_id": "go",
        "confidence": 0.8,
        "present_modalities": ["image", "sensor"],
        "options": [
            {"option_id": "stop", "attributes": {"action": "stop", "safe": True},
             "probability": 0.2},
            {"option_id": "go", "attributes": {"action": "go", "safe": False},
             "probability": 0.8},
        ],
    }
    assert json.loads(json.dumps(result)) == result


def test_native_json_abstention_does_not_emit_a_fake_distribution():
    result = native_decision_json(request(), None, abstain_reason="missing_required_signal")

    assert result["status"] == "abstain"
    assert result["selected_option_id"] is None
    assert result["confidence"] is None
    assert result["reason"] == "missing_required_signal"
    assert [option["probability"] for option in result["options"]] == [None, None]


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [
        ([0.2], None),
        ([0.2, 0.2], None),
        ([0.2, 0.9], None),
        ([float("nan"), 1.0], None),
        (None, None),
        ([0.5, 0.5], "uncertain"),
    ],
)
def test_native_json_rejects_invalid_probability_or_abstention_state(probabilities, reason):
    with pytest.raises(ValueError):
        native_decision_json(request(), probabilities, abstain_reason=reason)
