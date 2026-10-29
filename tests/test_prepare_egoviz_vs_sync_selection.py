from __future__ import annotations

import hashlib

from scripts import prepare_egoviz_vs_sync_selection as selection


def _fixture_records() -> tuple[list[dict[str, object]], set[str], set[str]]:
    groups = [f"group-{index:02}" for index in range(40)]
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256((selection.SPLIT_PREFIX + group).encode()).hexdigest(),
    )
    development = set(ordered[:8])
    training = set(ordered[8:])
    assignment = {
        group: selection.CATEGORIES[index // 2] for index, group in enumerate(ordered[:8])
    }
    for index, group in enumerate(ordered[8:]):
        assignment[group] = selection.CATEGORIES[index % len(selection.CATEGORIES)]

    records = []
    for number, (group, category) in enumerate(assignment.items()):
        records.append(
            {
                "category": category,
                "group_id": group,
                "mcap_bytes": 1000 + number,
                "chunk_id": f"chunk-{number:02}",
                "sequence_id": f"sequence-{number:02}",
                "mcap_path": f"data/{category}/chunk-{number:02}/chunk-{number:02}.mcap",
                "duration_s": 14.97,
                "n_frames": 450,
            }
        )
    return records, training, development


def test_selection_is_deterministic_and_group_disjoint():
    records, training, development = _fixture_records()
    split = {
        "development_group_sha256": sorted(
            hashlib.sha256(group.encode()).hexdigest() for group in development
        ),
        "train_group_sha256": sorted(
            hashlib.sha256(group.encode()).hexdigest() for group in training
        ),
    }
    checked_train, checked_development = selection._verify_split(records_group_ids(records), split)
    first, first_groups = selection._select_records(records, checked_train, checked_development)
    second, second_groups = selection._select_records(
        list(reversed(records)), checked_train, checked_development
    )

    assert first == second
    assert first_groups == second_groups
    assert len(first) == 16
    assert len(first_groups["train"]) == 8
    assert len(first_groups["development"]) == 8
    assert first_groups["train"].isdisjoint(first_groups["development"])


def records_group_ids(records: list[dict[str, object]]) -> list[str]:
    return [str(record["group_id"]) for record in records]
