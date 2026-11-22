"""Freeze a task-disjoint development split for the pinned HapTile conversion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

SOURCE_REVISION = "54887384f1fe9e93b799e104194a582f9d8840b8"
CONVERSION_REVISION = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
CONVERSION_CARD_SHA256 = "f0462ecc55d172e8a3c51ac5c874969e6f0482f093812482d28f730ba0558fbb"
SEED = 17
EXPECTED_TASKS = 38
EXPECTED_SAMPLES = 1699
SPLIT_SIZES = {"train": 24, "calibration": 4, "validation": 5, "sealed_audit": 5}
HASH_TEMPLATE = "tiny-omni-decision-haptile-task-v1|seed=17|task={task}"
_TASK_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_haptile_task_split(
    task_episode_counts: Mapping[str, int], *, conversion_card_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Return (public commitment, private full split) from task metadata only.

    The caller must provide per-task episode counts parsed from the pinned
    conversion card. This function never accepts media, actions, or labels.
    The full manifest is kept outside Git so sealed task names are not included
    in the public development record.
    """
    if conversion_card_sha256 != CONVERSION_CARD_SHA256:
        raise ValueError("HapTile conversion card hash differs from the pinned release")
    if len(task_episode_counts) != EXPECTED_TASKS:
        raise ValueError(f"expected exactly {EXPECTED_TASKS} task groups")

    normalized: dict[str, int] = {}
    for raw_task, count in task_episode_counts.items():
        if not isinstance(raw_task, str) or not _TASK_ID.fullmatch(raw_task):
            raise ValueError("task IDs must be canonical lowercase underscore slugs")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("each task group must have a positive integer episode count")
        normalized[raw_task] = count
    if len(normalized) != EXPECTED_TASKS or sum(normalized.values()) != EXPECTED_SAMPLES:
        raise ValueError("task inventory must contain exactly 1,699 converted episodes")

    ranked = sorted(
        normalized,
        key=lambda task: (
            hashlib.sha256(HASH_TEMPLATE.format(task=task).encode("ascii")).hexdigest(),
            task,
        ),
    )
    splits: dict[str, list[str]] = {}
    cursor = 0
    for split_name, group_count in SPLIT_SIZES.items():
        splits[split_name] = sorted(ranked[cursor:cursor + group_count])
        cursor += group_count

    task_counts = dict(sorted(normalized.items()))
    group_hashes = {
        task: hashlib.sha256(HASH_TEMPLATE.format(task=task).encode("ascii")).hexdigest()
        for task in sorted(normalized)
    }
    private_payload: dict[str, object] = {
        "format": "haptile_task_group_split_private_v1",
        "source_revision": SOURCE_REVISION,
        "conversion_revision": CONVERSION_REVISION,
        "conversion_card_sha256": CONVERSION_CARD_SHA256,
        "seed": SEED,
        "group_unit": "task slug; every converted episode in a task stays in one split",
        "hash_input_template": HASH_TEMPLATE,
        "split_sizes_in_task_groups": SPLIT_SIZES,
        "task_episode_counts": task_counts,
        "task_group_hashes": group_hashes,
        "splits": splits,
        "sealed_audit_accessed": False,
        "episode_media_or_action_labels_read": False,
    }
    private_payload["manifest_sha256"] = _canonical_sha256(private_payload)

    sealed_groups = splits["sealed_audit"]
    sealed_commitment = _canonical_sha256(
        {"groups": [{"task": task, "episodes": task_counts[task]} for task in sealed_groups]}
    )
    public_payload: dict[str, object] = {
        "format": "haptile_task_group_split_commitment_v1",
        "source_revision": SOURCE_REVISION,
        "conversion_revision": CONVERSION_REVISION,
        "conversion_card_sha256": CONVERSION_CARD_SHA256,
        "private_manifest_sha256": private_payload["manifest_sha256"],
        "seed": SEED,
        "group_unit": "task slug; no task appears across splits",
        "hash_input_template": HASH_TEMPLATE,
        "split_sizes_in_task_groups": SPLIT_SIZES,
        "development_splits": {
            name: splits[name] for name in ("train", "calibration", "validation")
        },
        "development_episode_counts": {
            name: sum(task_counts[task] for task in splits[name])
            for name in ("train", "calibration", "validation")
        },
        "sealed_audit": {
            "task_group_count": len(sealed_groups),
            "episode_count": sum(task_counts[task] for task in sealed_groups),
            "group_and_count_commitment_sha256": sealed_commitment,
            "task_ids_in_public_manifest": False,
        },
        "sealed_audit_accessed": False,
        "episode_media_or_action_labels_read": False,
    }
    public_payload["manifest_sha256"] = _canonical_sha256(public_payload)
    return public_payload, private_payload
