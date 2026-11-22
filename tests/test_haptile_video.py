import pytest

from tiny_omni_decision.haptile_video import select_causal_frame_indices


def test_select_causal_frame_indices_samples_even_past_window():
    times = [index / 10 for index in range(13)]
    result = select_causal_frame_indices(times, 1.0)
    assert result == (0, 1, 3, 4, 6, 7, 9, 10)
    assert all(times[index] <= 1.0 for index in result)


def test_select_causal_frame_indices_never_uses_future_frame():
    times = [index / 10 for index in range(15)]
    result = select_causal_frame_indices(times, 1.0, frame_count=4)
    assert max(times[index] for index in result) <= 1.0


def test_select_causal_frame_indices_assigns_distinct_frames_jointly():
    query = 1_000_000_016.852144
    times = [
        1_000_000_015.8159211,
        1_000_000_015.897723,
        1_000_000_015.975559,
        1_000_000_016.0522211,
        1_000_000_016.1365689,
        1_000_000_016.213207,
        1_000_000_016.2881991,
        1_000_000_016.36112,
        1_000_000_016.4329159,
        1_000_000_016.506926,
        1_000_000_016.580903,
        1_000_000_016.653687,
        1_000_000_016.7567921,
    ]

    result = select_causal_frame_indices(times, query)

    assert len(result) == len(set(result)) == 8
    targets = [query - 1 + index / 7 for index in range(8)]
    assert all(abs(times[index] - target) <= 0.1 for index, target in zip(
        result, targets, strict=True
    ))


@pytest.mark.parametrize(
    ("times", "query", "kwargs"),
    [
        ([0.0, 0.0, 1.0], 1.0, {}),
        ([float("nan"), 0.5, 1.0], 1.0, {}),
        ([0.0, 0.1], 1.0, {}),
        ([0.0, 1.0], 1.0, {"frame_count": 1}),
    ],
)
def test_select_causal_frame_indices_fails_closed(times, query, kwargs):
    with pytest.raises(ValueError):
        select_causal_frame_indices(times, query, **kwargs)
