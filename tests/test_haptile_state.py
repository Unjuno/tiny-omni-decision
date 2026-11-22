from __future__ import annotations

import pytest

from tiny_omni_decision.haptile_state import STREAM_WIDTHS, causal_state_sequence


def _streams(times=(0.0, 0.5, 1.0)):
    return {
        name: (times, [[float(index)] * width for index in range(len(times))])
        for name, width in STREAM_WIDTHS.items()
    }


def test_causal_state_sequence_uses_only_asof_observations():
    result = causal_state_sequence(_streams(), 1.0, context_seconds=1.0, steps=3)

    assert result.shape == (3, 25)
    assert result[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_causal_state_sequence_fails_when_a_stream_is_stale():
    streams = _streams(times=(0.0, 0.25, 0.5))

    with pytest.raises(ValueError, match="staleness"):
        causal_state_sequence(streams, 1.0, context_seconds=0.5, steps=3)


def test_causal_state_sequence_requires_every_declared_stream():
    streams = _streams()
    streams.pop("tactile")

    with pytest.raises(ValueError, match="all declared"):
        causal_state_sequence(streams, 1.0)
