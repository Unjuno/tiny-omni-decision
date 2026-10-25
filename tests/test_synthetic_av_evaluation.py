import numpy as np
import pytest

from scripts.evaluate_synthetic_av_readouts import _audio_shuffle_indices
from tiny_omni_decision.synthetic_av_evaluation import (
    group_accuracy,
    paired_group_bootstrap_difference,
    replay_prediction_metrics,
)
from tiny_omni_decision.synthetic_av_task import build_manifest


def test_group_accuracy_aggregates_examples_with_equal_group_weight():
    result = group_accuracy(
        np.asarray([1, 1, 0, 0, 0, 0]),
        np.asarray(["a", "a", "b", "b", "b", "b"]),
    )
    assert result == {"a": 1.0, "b": 0.0}


def test_paired_group_bootstrap_is_deterministic_and_checks_group_alignment():
    reference = {f"g{index}": 0.5 for index in range(16)}
    candidate = {f"g{index}": 0.75 for index in range(16)}
    first = paired_group_bootstrap_difference(reference, candidate, replicates=500, seed=17)
    assert first == paired_group_bootstrap_difference(
        reference, candidate, replicates=500, seed=17
    )
    assert first["mean_accuracy_difference"] == 0.25
    assert first["lower_95_percentile"] == first["upper_95_percentile"] == 0.25
    with pytest.raises(ValueError, match="same non-empty group IDs"):
        paired_group_bootstrap_difference(reference, {"other": 0.2})


def test_audio_pairing_control_shuffles_only_within_group_and_query_polarity():
    manifest = build_manifest(train_groups=1, validation_groups=2, seed=17)
    validation_rows = [row for row in manifest["examples"] if row["split"] == "validation"]
    arrays = {
        "example_ids": np.asarray([row["sample_id"] for row in manifest["examples"]]),
    }
    validation_indices = np.asarray(
        [index for index, row in enumerate(manifest["examples"]) if row["split"] == "validation"]
    )
    source_indices = _audio_shuffle_indices(arrays, manifest, validation_indices)
    source_ids = arrays["example_ids"][source_indices].tolist()
    source_rows = {row["sample_id"]: row for row in validation_rows}
    for index, source_id in zip(validation_indices, source_ids, strict=True):
        target_row = source_rows[arrays["example_ids"][index]]
        source_row = source_rows[source_id]
        assert source_row["group_id"] == target_row["group_id"]
        assert source_row["query_polarity"] == target_row["query_polarity"]
        assert source_row["pairing"] != target_row["pairing"]


def test_saved_prediction_replay_recomputes_pair_and_group_metrics():
    records = [
        {
            "probabilities": [0.9, 0.1],
            "target": 0,
            "group_id": "g0",
            "pair_id": "p0",
        },
        {
            "probabilities": [0.1, 0.9],
            "target": 1,
            "group_id": "g0",
            "pair_id": "p0",
        },
    ]
    metrics = replay_prediction_metrics(records)
    assert metrics["accuracy"] == 1.0
    assert metrics["pair_exact_accuracy"] == 1.0
    assert metrics["group_macro_accuracy"] == 1.0
    assert metrics["group_count"] == 1
