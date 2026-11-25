import pytest

from tiny_omni_decision.robopro_actions import (
    causal_window_indices,
    numeric_option_separation,
    uniform_indices,
    validate_probe_feature_shapes,
)


def test_uniform_indices_cover_clip_endpoints():
    assert uniform_indices(169, 8) == [0, 24, 48, 72, 96, 120, 144, 168]


@pytest.mark.parametrize("query_index", [42, 90, 168])
def test_causal_window_is_distinct_and_ends_at_query(query_index):
    indices = causal_window_indices(169, query_index)

    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert indices[-1] == query_index
    assert all(0 <= index <= query_index for index in indices)
    assert indices[-1] - indices[0] <= 100


def test_causal_window_rejects_too_short_past():
    with pytest.raises(ValueError, match="too little past context"):
        causal_window_indices(169, 5)


def test_numeric_option_separation_reports_duplicate_choices():
    result = numeric_option_separation(
        [[0.0, 1.0], [0.0, 1.0], [0.0000005, 1.0000003], [3.0, 4.0]],
        tolerance=1e-6,
    )

    assert result["dimension"] == 2
    assert result["pair_count"] == 6
    assert result["exact_duplicate_pairs"] == [[0, 1]]
    assert result["within_tolerance_pairs"] == [[0, 1], [0, 2], [1, 2]]


@pytest.mark.parametrize(
    "options",
    [
        [[1.0], [float("nan")]],
        [[1.0], [float("inf")]],
        [[1.0], [1.0, 2.0]],
        [[], []],
    ],
)
def test_numeric_option_separation_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        numeric_option_separation(options)


def test_probe_feature_shapes_follow_selected_sample_size():
    validate_probe_feature_shapes(
        (512, 3, 8, 768),
        (512, 3, 8, 14),
        (512, 3, 14),
        (128, 3, 4, 14),
        episode_count=512,
        prompt_group_count=128,
    )


def test_probe_feature_shapes_reject_mismatched_episode_count():
    with pytest.raises(ValueError, match="unexpected temporal feature shape"):
        validate_probe_feature_shapes(
            (128, 3, 8, 768),
            (512, 3, 8, 14),
            (512, 3, 14),
            (128, 3, 4, 14),
            episode_count=512,
            prompt_group_count=128,
        )
