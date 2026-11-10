import pytest

pytest.importorskip("torch", reason="optional Path A model-evaluation dependency")

import numpy as np

from scripts.evaluate_perception_test_video_readout import (
    _paired_group_bootstrap,
    _require_fixed_validation_split,
    donor_video_map,
)
from scripts.train_perception_test_video_readout import _metrics


def test_donor_map_is_deterministic_and_never_uses_same_video():
    ids = [f"video-{index}" for index in range(12)]

    first = donor_video_map(ids, seed=17)
    second = donor_video_map(list(reversed(ids)), seed=17)

    assert first == second
    assert all(video_id != donor for video_id, donor in first.items())
    assert set(first) == set(first.values())


def test_validation_reader_uses_frozen_corpus_split_value():
    _require_fixed_validation_split([{"split": "valid"}, {"split": "valid"}])

    with pytest.raises(ValueError, match="unexpected split"):
        _require_fixed_validation_split([{"split": "validation"}])


def test_paired_group_bootstrap_is_deterministic_and_reports_direction():
    targets = np.asarray([0, 1, 0, 1, 0, 1])
    groups = ["a", "a", "b", "b", "c", "c"]
    joint = np.asarray([[4, 0], [0, 4], [4, 0], [0, 4], [4, 0], [0, 4]], dtype=float)
    text = np.zeros_like(joint)
    donor = np.full_like(joint, -1.0)

    first = _paired_group_bootstrap(
        {"joint": joint, "text_only": text, "mismatched_video": donor},
        targets,
        groups,
        repetitions=100,
    )
    second = _paired_group_bootstrap(
        {"joint": joint, "text_only": text, "mismatched_video": donor},
        targets,
        groups,
        repetitions=100,
    )

    assert first == second
    assert first["text_only"]["metrics"]["accuracy"]["delta"] > 0
    assert first["text_only"]["metrics"]["nll"]["delta"] < 0


def test_uniform_three_choice_scores_have_expected_calibration_metrics():
    result = _metrics(np.zeros((3, 3)), np.asarray([0, 1, 2]))

    assert result["accuracy"] == 1 / 3
    assert abs(result["nll"] - np.log(3)) < 1e-12
    assert abs(result["brier"] - 2 / 3) < 1e-12
    assert result["ece_15_bins"] == 0
