import json

import pytest

from scripts.check_cohort_usage import group_digest
from scripts.cohort_gate import CohortUseBlocked, require_clear_cohort

SOURCE = "example/dataset"
REVISION = "a" * 40


def _write_ledger(path, *, status="COMPLETE", used=()):
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": SOURCE,
                        "revision": REVISION,
                        "history_status": status,
                        "usage": [
                            {
                                "purpose": "prior_run",
                                "group_identity_sha256": [
                                    group_digest(SOURCE, REVISION, item) for item in used
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_cohort_gate_allows_new_groups_only_with_complete_history(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    _write_ledger(ledger_path)

    result = require_clear_cohort(
        ledger_path=ledger_path,
        source_id=SOURCE,
        revision=REVISION,
        group_ids=["new-1", "new-2"],
    )

    assert result["decision"] == "CLEAR_WITHIN_DECLARED_HISTORY"
    assert result["proposed_group_count"] == 2
    assert result["proposed_group_identity_sha256"] == sorted(
        [group_digest(SOURCE, REVISION, "new-1"), group_digest(SOURCE, REVISION, "new-2")]
    )
    assert len(result["ledger_sha256"]) == 64


@pytest.mark.parametrize(
    ("status", "groups", "decision"),
    [
        ("COMPLETE", ["used"], "BLOCKED_PRIOR_GROUP_USE"),
        ("INCOMPLETE", ["new"], "BLOCKED_INCOMPLETE_HISTORY"),
    ],
)
def test_cohort_gate_blocks_overlap_or_incomplete_history(tmp_path, status, groups, decision):
    ledger_path = tmp_path / "ledger.json"
    _write_ledger(ledger_path, status=status, used=["used"])

    with pytest.raises(CohortUseBlocked, match=decision):
        require_clear_cohort(
            ledger_path=ledger_path,
            source_id=SOURCE,
            revision=REVISION,
            group_ids=groups,
        )


def test_cohort_gate_blocks_unknown_revision(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    _write_ledger(ledger_path)

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        require_clear_cohort(
            ledger_path=ledger_path,
            source_id=SOURCE,
            revision="b" * 40,
            group_ids=["new"],
        )
