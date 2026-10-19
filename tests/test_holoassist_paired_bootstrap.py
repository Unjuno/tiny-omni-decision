from __future__ import annotations

import pytest

from scripts.analyze_holoassist_paired_bootstrap import paired_session_bootstrap


def _rows(offset: float = 0.0):
    rows = {}
    for session in range(4):
        for window in range(2):
            target = (session + window) % 2
            positive_probability = 0.8 if target else 0.2
            positive_probability = min(0.99, max(0.01, positive_probability + offset))
            rows[f"s{session}:w{window}"] = {
                "session_id": f"s{session}",
                "target": target,
                "probabilities": [1 - positive_probability, positive_probability],
            }
    return rows


def test_paired_session_bootstrap_is_deterministic_and_preserves_cluster_count():
    result = paired_session_bootstrap(
        _rows(), _rows(-0.05), replicates=200, seed=1701
    )
    repeated = paired_session_bootstrap(
        _rows(), _rows(-0.05), replicates=200, seed=1701
    )

    assert result == repeated
    assert result["session_count"] == 4
    assert result["window_count"] == 8
    assert result["candidate_minus_baseline"]["nll"]["valid_replicates"] == 200
    assert result["candidate_minus_baseline"]["balanced_accuracy"]["valid_replicates"] <= 200


def test_paired_session_bootstrap_rejects_nonmatching_validation_ids():
    candidate = _rows()
    candidate.pop("s0:w0")
    with pytest.raises(ValueError, match="prediction IDs must match exactly"):
        paired_session_bootstrap(_rows(), candidate, replicates=10)
