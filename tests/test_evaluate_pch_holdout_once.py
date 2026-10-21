import numpy as np
import pytest

from scripts.evaluate_pch_holdout_once import (
    aggregate_window_logits,
    paired_subject_bootstrap,
    summarize_logits,
    verify_content_separation,
    verify_holdout_members,
)


def test_aggregate_window_logits_averages_within_recording_only():
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    result = aggregate_window_logits(logits, np.asarray([0, 0, 1]), record_count=2)
    np.testing.assert_array_equal(result, np.asarray([[1.0, 1.0], [4.0, 0.0]], dtype=np.float32))


def test_verify_holdout_members_requires_frozen_subject_set_and_no_other_split():
    members = {
        "fit": [{"subject_id": "P001"}],
        "validation": [{"subject_id": "P002"}],
        "heldout": [{"subject_id": "P003"}, {"subject_id": "P004"}],
    }
    result = verify_holdout_members(
        members,
        expected_subject_ids=["P003", "P004"],
        expected_subject_hash=None,
    )
    assert result["record_count"] == 2
    assert result["subject_ids"] == ["P003", "P004"]

    members["fit"].append({"subject_id": "P003"})
    with pytest.raises(ValueError, match="overlap"):
        verify_holdout_members(
            members,
            expected_subject_ids=["P003", "P004"],
            expected_subject_hash=None,
        )


def test_summarize_logits_reports_fixed_class_metrics():
    result = summarize_logits(
        np.asarray([[4.0, 0.0], [0.0, 4.0]]), np.asarray([0, 1]), class_count=2
    )
    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy"] == 1.0
    assert result["nll"] < 0.1
    assert len(result["per_class"]) == 2


def test_content_overlap_rejects_duplicate_or_cross_split_media_identity():
    verify_content_separation(["h1"], ["s1"], ["f1"], ["v1"])
    with pytest.raises(ValueError, match="raw content overlap"):
        verify_content_separation(["f1"], ["s1"], ["f1"], ["v1"])
    with pytest.raises(ValueError, match="normalized signal overlap"):
        verify_content_separation(["h1"], ["v1"], ["f1"], ["v1"])


def test_paired_subject_bootstrap_is_deterministic_and_reports_direction():
    targets = np.asarray([0, 0, 1, 1])
    logits_a = np.asarray([[3, 0], [0, 3], [0, 3], [0, 3]], dtype=np.float64)
    logits_b = np.asarray([[3, 0], [3, 0], [0, 3], [0, 3]], dtype=np.float64)
    subjects = np.asarray(["P1", "P1", "P2", "P2"])
    first = paired_subject_bootstrap(logits_a, logits_b, targets, subjects, draws=200, seed=17)
    second = paired_subject_bootstrap(logits_a, logits_b, targets, subjects, draws=200, seed=17)
    assert first == second
    assert first["direction"] == "model_a_minus_model_b"
    assert first["cluster_count"] == 2
