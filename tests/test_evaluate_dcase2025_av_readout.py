from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.evaluate_dcase2025_av_readout import (
    cluster_bootstrap_differences,
    parent_derangement,
)


def test_parent_derangement_moves_every_sample_to_a_different_recording():
    parents = ["a", "a", "b", "b", "c", "c"]

    donors = parent_derangement(parents, seed=17)

    assert len(donors) == len(parents)
    assert all(parents[index] != parents[donor] for index, donor in enumerate(donors))


def test_cluster_bootstrap_is_paired_and_reports_low_group_power():
    targets = [0, 1, 0, 1]
    parents = ["p1", "p1", "p2", "p2"]
    rooms = ["r1", "r1", "r2", "r2"]
    probabilities = {
        "audio": np.asarray([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.4, 0.6]]),
        "video": np.asarray([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]]),
        "joint": np.asarray([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.2, 0.8]]),
    }

    result = cluster_bootstrap_differences(
        targets, parents, rooms, probabilities, replicates=32, seed=17
    )

    assert result["parent_recording"]["independent_groups"] == 2
    assert result["room"]["low_power_under_200_groups"] is True
    difference = result["parent_recording"]["comparisons"]["joint_minus_audio"]
    assert difference["accuracy_difference"] == 0.25
    assert len(difference["percentile_95_ci"]["nll_difference"]) == 2
