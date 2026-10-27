import copy

import pytest

pytest.importorskip("torch")
from scripts.audit_source_confidence import audit  # noqa: E402


def row(correct, confidence, nll):
    return {
        "correct": correct,
        "confidence": confidence,
        "nll": nll,
        "target_candidate_ids": ["a", "b"],
        "candidate_ids": ["a", "b", "c"],
        "none_visible": False,
        "video_id": "video1",
    }


def test_confidence_counts_and_paired_transitions():
    data = {
        4: {"x": row(False, 0.95, 3.0), "y": row(True, 0.6, 0.1)},
        8: {"x": row(True, 0.6, 0.1), "y": row(False, 0.4, 2.0)},
        32: {"x": row(True, 0.6, 0.0), "y": row(True, 0.5, 0.0)},
    }
    result = audit(data)
    assert result["epochs"][4]["errors_confidence_gt_0_9"] == 1
    assert result["epochs"][4]["target_count"][2] == 2
    assert result["transitions"]["4-8"] == {"both_wrong": 0, "fixed": 1, "new_errors": 1}
    assert result["epochs"][32]["mean_error_nll"] is None
    assert result["epochs"][32]["error_nll_fraction"] is None


def test_reject_misaligned_targets_and_records():
    data = {epoch: {"x": row(False, 0.9, 1.0)} for epoch in (4, 8, 32)}
    bad = copy.deepcopy(data)
    bad[32]["x"]["target_candidate_ids"] = ["c"]
    with pytest.raises(ValueError, match="target mismatch"):
        audit(bad)
    bad = copy.deepcopy(data)
    bad[8] = {"y": bad[8]["x"]}
    with pytest.raises(ValueError, match="record IDs"):
        audit(bad)
