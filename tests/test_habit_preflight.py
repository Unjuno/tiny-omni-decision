from __future__ import annotations

from tiny_omni_decision.habit_preflight import build_habit_grouped_split


def _episode(index: int, sid: str, instruction: str, task_index: int) -> dict:
    return {
        "episode_index": index,
        "sid": sid,
        "high_level_instruction": instruction,
        "task_index": task_index,
    }


def test_habit_split_keeps_scenario_aliases_and_instruction_duplicates_together(
    tmp_path,
):
    details = tmp_path / "task_details"
    details.mkdir()
    for name, role in [
        ("S1_S2", "Collaborator"),
        ("S3", "Collaborator"),
        ("S4", "Coworker"),
        ("S5", "Coworker"),
        ("S6", "Supervisor"),
        ("S7", "Supervisor"),
    ]:
        (details / f"{name}.md").write_text(f"# Task\n\n## {role}\n", encoding="utf-8")

    rows = [
        _episode(1, "S1", "Clean the shelf.", 1),
        _episode(2, "S2", "Clean the shelf.", 1),
        _episode(3, "S3", " clean   the shelf. ", 2),
        _episode(4, "S4", "Pack items.", 3),
        _episode(5, "S5", "Pack items variant.", 3),
        _episode(6, "S6", "Move object.", 4),
        _episode(7, "S7", "Move object variant.", 5),
        _episode(8, "S8", "Undocumented task.", 6),
    ]
    result = build_habit_grouped_split(rows, details, seed=17)

    train = set(result["train_episode_ids"])
    validation = set(result["validation_episode_ids"])
    assert {1, 2, 3} <= train | validation
    assert not ({1, 2, 3} & train and {1, 2, 3} & validation)
    assert not ({4, 5} & train and {4, 5} & validation)
    assert result["quarantined_episode_count"] == 1
    assert result["sid_without_task_details"] == ["S8"]
    assert result["sealed_audit_episode_ids"] is None


def test_habit_split_is_deterministic_and_does_not_emit_audit_ids(tmp_path):
    details = tmp_path / "task_details"
    details.mkdir()
    rows = []
    for role_index, role in enumerate(("Collaborator", "Coworker", "Supervisor")):
        for group_index in range(8):
            sid = f"S{role_index}{group_index}"
            (details / f"{sid}.md").write_text(
                f"# Task {sid}\n\n## {role}\n", encoding="utf-8"
            )
            rows.append(_episode(len(rows), sid, f"Task {sid}.", role_index * 100 + group_index))

    first = build_habit_grouped_split(rows, details, seed=17)
    second = build_habit_grouped_split(rows, details, seed=17)
    assert first == second
    assert first["sealed_audit_episode_ids"] is None
    assert first["sealed_audit_episode_count"] > 0
    assert set(first["train_episode_ids"]).isdisjoint(first["validation_episode_ids"])
    assert len(first["train_episode_ids"]) + len(first["validation_episode_ids"]) < len(rows)


def test_habit_role_typo_is_normalized_and_cross_role_text_is_train_only(tmp_path):
    details = tmp_path / "task_details"
    details.mkdir()
    (details / "S1.md").write_text("# Task\n\n## Colaborator\n", encoding="utf-8")
    (details / "S2.md").write_text("# Task\n\n## Supervisor\n", encoding="utf-8")
    rows = [
        _episode(1, "S1", "Same instruction.", 1),
        _episode(2, "S2", "same instruction.", 2),
    ]

    result = build_habit_grouped_split(rows, details, seed=17)
    assert result["group_counts_by_split"] == {"train": 1}
    assert result["train_episode_ids"] == [1, 2]
    assert result["validation_episode_ids"] == []


def test_habit_preflight_rejects_duplicate_episode_ids(tmp_path):
    details = tmp_path / "task_details"
    details.mkdir()
    (details / "S1.md").write_text("# Task\n\n## Collaborator\n", encoding="utf-8")
    rows = [_episode(1, "S1", "Task.", 1), _episode(1, "S1", "Task.", 1)]

    try:
        build_habit_grouped_split(rows, details, seed=17)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate episode IDs must fail closed")
