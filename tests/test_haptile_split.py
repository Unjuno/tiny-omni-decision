import hashlib
import json

import pytest

from tiny_omni_decision.haptile_split import (
    HASH_TEMPLATE,
    SPLIT_SIZES,
    build_haptile_task_split,
)

CARD_SHA256 = "f0462ecc55d172e8a3c51ac5c874969e6f0482f093812482d28f730ba0558fbb"


def sample_task_counts():
    counts = {f"task_{index:02d}": 44 for index in range(1, 38)}
    counts["task_38"] = 71
    return counts


def test_haptile_task_split_is_deterministic_disjoint_and_order_independent():
    counts = sample_task_counts()
    first = build_haptile_task_split(counts, conversion_card_sha256=CARD_SHA256)
    second = build_haptile_task_split(dict(reversed(list(counts.items()))),
                                      conversion_card_sha256=CARD_SHA256)
    public, private = first
    assert first == second
    assert {name: len(groups) for name, groups in private["splits"].items()} == SPLIT_SIZES
    groups = [task for values in private["splits"].values() for task in values]
    assert len(groups) == len(set(groups)) == 38
    assert public["sealed_audit"]["task_ids_in_public_manifest"] is False
    assert "sealed_audit" not in public["development_splits"]
    assert public["sealed_audit_accessed"] is False
    assert private["episode_media_or_action_labels_read"] is False


def test_haptile_task_hash_and_manifest_commitments_are_reproducible():
    public, private = build_haptile_task_split(
        sample_task_counts(), conversion_card_sha256=CARD_SHA256
    )
    any_task = next(iter(private["task_group_hashes"]))
    assert private["task_group_hashes"][any_task] == hashlib.sha256(
        HASH_TEMPLATE.format(task=any_task).encode("ascii")
    ).hexdigest()
    public_hash = public.pop("manifest_sha256")
    private_hash = private.pop("manifest_sha256")
    assert hashlib.sha256(json.dumps(
        public, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()).hexdigest() == public_hash
    assert hashlib.sha256(json.dumps(
        private, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()).hexdigest() == private_hash


@pytest.mark.parametrize("counts,card_sha", [
    ({f"task_{index:02d}": 44 for index in range(1, 38)}, CARD_SHA256),
    ({**sample_task_counts(), "task_39": 1}, CARD_SHA256),
    ({**sample_task_counts(), "BAD TASK": 1}, CARD_SHA256),
    ({**sample_task_counts(), "task_38": 0}, CARD_SHA256),
    (sample_task_counts(), "0" * 64),
])
def test_invalid_haptile_inventory_or_revision_fails_closed(counts, card_sha):
    with pytest.raises(ValueError):
        build_haptile_task_split(counts, conversion_card_sha256=card_sha)
