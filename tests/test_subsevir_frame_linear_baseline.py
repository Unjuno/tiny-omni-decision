import numpy as np
import pytest

from scripts.evaluate_subsevir_frame_linear_baseline import (
    compute_metrics,
    group_bootstrap,
    load_rows,
)


def test_compute_metrics_uses_probability_scores_and_binary_confusion():
    targets = np.asarray([0, 1], dtype=np.int64)
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)

    result = compute_metrics(targets, probabilities)

    assert result["accuracy"] == 1.0
    assert result["correct_count"] == 2
    assert result["confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert result["nll"] == pytest.approx((-np.log(0.8) - np.log(0.7)) / 2)
    assert result["brier"] == pytest.approx((0.2**2 + 0.2**2 + 0.3**2 + 0.3**2) / 2)


def test_group_bootstrap_is_deterministic_and_paired_by_event():
    targets = np.asarray([0, 1, 1], dtype=np.int64)
    current = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    reference = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    groups = ["scene-a", "scene-a", "scene-b"]

    first = group_bootstrap(targets, current, reference, groups)
    second = group_bootstrap(targets, current, reference, groups)

    assert first == second
    assert first["unit"] == "event_id"
    assert first["group_count"] == 2
    assert first["replicates"] == 10_000
    assert "difference_vs_joint_lstm_accuracy" in first


def test_load_rows_uses_only_the_last_preceding_frame_and_preserves_group():
    manifest = {
        "windows": [
            {
                "split": "train",
                "sample_id": "event-a:target-8",
                "event_id": "event-a",
                "target": 1,
                "input_indices": list(range(8)),
            },
            {
                "split": "validation",
                "sample_id": "event-b:target-18",
                "event_id": "event-b",
                "target": 0,
                "input_indices": list(range(10, 18)),
            },
        ]
    }
    frame_indices = np.asarray([7, 17], dtype=np.int64)
    features = np.asarray(
        [
            [[7.0] * 768, [-7.0] * 768],
            [[17.0] * 768, [-17.0] * 768],
        ],
        dtype=np.float32,
    )

    rows = load_rows(manifest, frame_indices, features)

    assert rows["train"][0]["event_id"] == "event-a"
    assert rows["train"][0]["feature"].shape == (1_536,)
    assert rows["train"][0]["feature"][0] == 7.0
    assert rows["train"][0]["feature"][768] == -7.0
    assert rows["validation"][0]["feature"][0] == 17.0
