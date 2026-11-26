import pytest

np = pytest.importorskip("numpy")

from scripts.evaluate_dcase2025_av_match_zero_shot import (  # noqa: E402
    cosine_probabilities,
    metrics,
    paired_bootstrap,
)


def test_direct_cosine_prefers_the_matching_candidate_without_fitting():
    audio = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    candidates = np.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]],
        ]
    )

    probabilities = cosine_probabilities(audio, candidates)

    assert probabilities.argmax(1).tolist() == [1, 1]
    assert np.allclose(probabilities.sum(1), 1.0)
    assert metrics(probabilities, np.asarray([1, 1]))["accuracy"] == 1.0


def test_direct_cosine_rejects_misaligned_or_nonfinite_features():
    with pytest.raises(ValueError, match="aligned"):
        cosine_probabilities(np.ones((2, 3)), np.ones((1, 4, 3)))
    with pytest.raises(ValueError, match="finite"):
        cosine_probabilities(np.asarray([[np.nan, 0.0]]), np.ones((1, 4, 2)))


def test_paired_bootstrap_preserves_parent_and_room_groups():
    targets = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    probabilities = np.full((8, 4), 0.1)
    probabilities[np.arange(8), targets] = 0.7
    uniform = np.full((8, 4), 0.25)

    result = paired_bootstrap(
        probabilities,
        uniform,
        targets,
        ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"],
        ["r0"] * 4 + ["r1"] * 4,
        replicates=64,
        seed=17,
    )

    assert result["parent_recording"]["groups"] == 4
    assert result["room"]["groups"] == 2
    assert result["parent_recording"]["point"]["accuracy"] == pytest.approx(0.75)
