import json

import pytest

from scripts.check_cohort_usage import check_cohort, group_digest, read_group_ids

SOURCE = "google-deepmind/perception_test"
REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"


def _ledger(*, history_status="INCOMPLETE"):
    return {
        "sources": [
            {
                "source_id": SOURCE,
                "revision": REVISION,
                "history_status": history_status,
                "usage": [
                    {
                        "purpose": "validation_evaluation",
                        "group_identity_sha256": [group_digest(SOURCE, REVISION, "video_1")],
                    }
                ],
            }
        ]
    }


def test_blocks_previously_used_group_by_source_scoped_identity():
    result = check_cohort(
        _ledger(history_status="COMPLETE"),
        source_id=SOURCE,
        revision=REVISION,
        group_ids=["video_1"],
    )
    assert result["decision"] == "BLOCKED_PRIOR_GROUP_USE"
    assert result["overlap_count"] == 1


def test_incomplete_history_blocks_unseen_group():
    result = check_cohort(
        _ledger(), source_id=SOURCE, revision=REVISION, group_ids=["video_new"]
    )
    assert result["decision"] == "BLOCKED_INCOMPLETE_HISTORY"
    assert result["overlap_count"] == 0


def test_complete_history_allows_only_nonoverlapping_group():
    result = check_cohort(
        _ledger(history_status="COMPLETE"),
        source_id=SOURCE,
        revision=REVISION,
        group_ids=["video_new"],
    )
    assert result["decision"] == "CLEAR_WITHIN_DECLARED_HISTORY"


def test_unknown_source_or_revision_fails_closed():
    result = check_cohort(
        _ledger(history_status="COMPLETE"),
        source_id=SOURCE,
        revision="unknown-revision",
        group_ids=["video_new"],
    )
    assert result["decision"] == "BLOCKED_UNKNOWN_SOURCE_OR_REVISION"


def test_rejects_duplicate_or_empty_proposed_ids():
    with pytest.raises(ValueError, match="unique"):
        check_cohort(_ledger(), source_id=SOURCE, revision=REVISION, group_ids=["v", "v"])
    with pytest.raises(ValueError, match="unique"):
        check_cohort(_ledger(), source_id=SOURCE, revision=REVISION, group_ids=[])


def test_reads_jsonl_ids_and_deduplicates_repeated_group_rows(tmp_path):
    path = tmp_path / "groups.jsonl"
    path.write_text(
        '{"video_id":"v1"}\n{"video_id":"v1"}\n{"video_id":"v2"}\n',
        encoding="utf-8",
    )
    assert read_group_ids(path, "video_id") == ["v1", "v2"]


def test_reads_json_array_rejects_duplicate_group_ids(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(["v1", "v1"]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_group_ids(path, "unused")


def test_reads_json_array(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(["v1", "v2"]), encoding="utf-8")
    assert read_group_ids(path, "unused") == ["v1", "v2"]
