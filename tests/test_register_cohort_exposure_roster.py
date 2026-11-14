from __future__ import annotations

import copy

import pytest

from scripts.check_cohort_usage import check_cohort, group_digest
from scripts.register_cohort_exposure_roster import register_roster


def _ledger() -> dict:
    return {
        "format": "pretrained_reuse_cohort_usage_ledger_v1",
        "generated_from": "old evidence",
        "sources": [],
    }


def _roster() -> dict:
    sources = []
    for source_id, revision in (
        ("castacks/TartanAviation", "official-rev"),
        ("twangodev/tartanaviation-atc-adsb-utterances", "utterance-rev"),
        ("twangodev/tartanaviation-atc-labels", "labels-rev"),
    ):
        hashes = [
            group_digest(source_id, revision, "kagc/01-02-22/0007"),
            group_digest(source_id, revision, "kbtp/02-03-22/0008"),
        ]
        sources.append(
            {
                "source_id": source_id,
                "revision": revision,
                "group_identity_kind": "parent_clip_id",
                "group_count": 2,
                "group_identity_sha256": hashes,
            }
        )
    return {
        "schema_version": 1,
        "exposure": "metadata_only_utterance_ids_read_no_media_or_content",
        "verified_shard_count": 184,
        "unique_parent_clip_count": 2,
        "sources": sources,
    }


def test_registers_known_groups_but_keeps_all_source_history_incomplete() -> None:
    result = register_roster(
        _ledger(), _roster(), roster_path="roster.json", roster_sha256="a" * 64
    )

    assert len(result["sources"]) == 3
    assert all(source["history_status"] == "INCOMPLETE" for source in result["sources"])
    assert all(
        len(source["usage"][0]["group_identity_sha256"]) == 2
        for source in result["sources"]
    )
    assert result["generated_from"].startswith("prior saved validation predictions")


def test_imported_group_hash_blocks_exact_group_and_incomplete_history_blocks_new_group() -> None:
    result = register_roster(
        _ledger(), _roster(), roster_path="roster.json", roster_sha256="a" * 64
    )
    source = result["sources"][0]
    exact = check_cohort(
        result,
        source_id=source["source_id"],
        revision=source["revision"],
        group_ids=["kagc/01-02-22/0007"],
    )
    fresh = check_cohort(
        result,
        source_id=source["source_id"],
        revision=source["revision"],
        group_ids=["kagc/03-04-22/0009"],
    )
    assert exact["decision"] == "BLOCKED_PRIOR_GROUP_USE"
    assert fresh["decision"] == "BLOCKED_INCOMPLETE_HISTORY"


def test_rejects_duplicate_source_revision_without_mutating_input() -> None:
    original = _ledger()
    once = register_roster(
        original, _roster(), roster_path="roster.json", roster_sha256="a" * 64
    )
    before = copy.deepcopy(once)
    with pytest.raises(ValueError, match="already exists"):
        register_roster(
            once, _roster(), roster_path="roster.json", roster_sha256="a" * 64
        )
    assert once == before


def test_rejects_roster_with_changed_history_or_bad_hashes() -> None:
    bad_history = _roster()
    bad_history["exposure"] = "model_or_media_used"
    with pytest.raises(ValueError, match="metadata-only exposure"):
        register_roster(
            _ledger(), bad_history, roster_path="roster.json", roster_sha256="a" * 64
        )

    bad_hash = _roster()
    bad_hash["sources"][0]["group_identity_sha256"][0] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        register_roster(
            _ledger(), bad_hash, roster_path="roster.json", roster_sha256="a" * 64
        )
