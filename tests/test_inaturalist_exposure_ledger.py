from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit_inaturalist_exposure_ledger import audit_cohorts, identity_digest


def _row(observation: str, photo: str, observer: str, split: str) -> dict[str, str]:
    return {
        "observation_id": observation,
        "photo_id": photo,
        "observer_group_sha256": observer,
        "label": "Mallard",
        "split": split,
    }


def test_identity_digest_matches_canonical_array_encoding() -> None:
    expected = hashlib.sha256(
        json.dumps(
            ["inaturalist.org", "observer_group_sha256", "group-a"],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert identity_digest("inaturalist.org", "observer_group_sha256", "group-a") == expected


def test_audit_flags_excluded_observers_missing_from_incomplete_ledger() -> None:
    prior = identity_digest("inaturalist.org", "observer_group_sha256", "old-group")
    ledger = {
        "source_id": "inaturalist.org",
        "history_status": "INCOMPLETE",
        "usage": [{"identity_hashes_by_kind": {"observer_group_sha256": [prior]}}],
        "comparison_metadata": {"previous_observer_count": 2},
    }
    source_train = [
        _row("old-observation", "old-photo", "old-group", "train"),
        _row("missing-history-observation", "missing-history-photo", "missing-group", "train"),
    ]
    retained_train = []

    result = audit_cohorts(ledger, source_train, [], retained_train, [])

    assert result["ledger_unique_observer_digests"] == 1
    assert result["preflight_minus_ledger_observer_count"] == 1
    assert result["excluded_rows_without_ledger_observer_match"] == 1
    assert result["unique_excluded_observer_groups_without_ledger_match"] == 1
    assert result["history_reconciled"] is False


def test_audit_accepts_a_fully_reconciled_disjoint_cohort() -> None:
    prior = identity_digest("inaturalist.org", "observer_group_sha256", "old-group")
    ledger = {
        "source_id": "inaturalist.org",
        "history_status": "COMPLETE",
        "usage": [{"identity_hashes_by_kind": {"observer_group_sha256": [prior]}}],
        "comparison_metadata": {"previous_observer_count": 1},
    }
    old = _row("old-observation", "old-photo", "old-group", "train")
    train = _row("new-train-observation", "new-train-photo", "new-train-group", "train")
    validation = _row(
        "new-validation-observation", "new-validation-photo", "new-validation-group", "validation"
    )

    result = audit_cohorts(ledger, [old, train], [validation], [train], [validation])

    assert result["excluded_rows"] == 1
    assert result["excluded_rows_without_ledger_observer_match"] == 0
    assert result["retained_train_validation_observer_overlap"] == 0
    assert result["history_reconciled"] is True


def test_audit_rejects_retained_rows_missing_from_frozen_source() -> None:
    ledger = {
        "source_id": "inaturalist.org",
        "history_status": "COMPLETE",
        "usage": [],
        "comparison_metadata": {"previous_observer_count": 0},
    }
    with pytest.raises(ValueError, match="not a subset"):
        audit_cohorts(
            ledger,
            [_row("source", "source-photo", "group-a", "train")],
            [],
            [_row("other", "other-photo", "group-b", "train")],
            [],
        )
