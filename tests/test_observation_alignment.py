import pytest

from tiny_omni_decision.observation_alignment import align_observations
from tiny_omni_decision.observations import ObservationBatch


def batch():
    return ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "options": [{"option_id": "a", "attributes": {}}, {"option_id": "b", "attributes": {}}],
            "observations": [
                {
                    "modality": "video",
                    "source_id": "v",
                    "payload_ref": "unused",
                    "payload_sha256": "a" * 64,
                    "clock_id": "device",
                    "timestamps_ms": [0, 10, 20],
                    "valid_mask": [True, False, True],
                },
                {
                    "modality": "audio",
                    "source_id": "a",
                    "payload_ref": "unused",
                    "payload_sha256": "b" * 64,
                    "clock_id": "device",
                    "timestamps_ms": [5, 15],
                    "valid_mask": [True, True],
                },
            ],
        }
    )


def test_causal_age_and_invalid_samples():
    aligned = align_observations(batch(), [0, 10, 20, 100], max_age_ms=12)
    assert aligned["v"]["indices"] == (0, 0, 2, None)
    assert aligned["a"]["indices"] == (None, 0, 1, None)
    assert aligned["a"]["available"] == (False, True, True, False)
    assert aligned["v"]["age_ms"] == (0, 10, 0, None)


def test_cross_clock_and_bad_queries_rejected():
    b = batch()
    changed = b.observations[1].model_copy(update={"clock_id": "other"})
    with pytest.raises(ValueError):
        align_observations(
            b.model_copy(update={"observations": (b.observations[0], changed)}), [10], max_age_ms=20
        )
    for queries in ([10, 0], [0, 0], [float("nan")], []):
        with pytest.raises(ValueError):
            align_observations(b, queries, max_age_ms=20)


def test_future_observation_is_never_exposed():
    result = align_observations(batch(), [-1], max_age_ms=1000)
    assert all(r["indices"] == (None,) for r in result.values())
