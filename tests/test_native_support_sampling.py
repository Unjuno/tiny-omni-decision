import copy
from collections import Counter

import pytest


def fixture():
    rows = []
    for i, (source, task, target) in enumerate([
        ("starss", "clapping", 0), ("starss", "clapping", 0),
        ("starss", "clapping", 0), ("starss", "clapping", 0),
        ("starss", "speech", 1), ("starss", "speech", 0),
        ("text", "choose", 0), ("text", "choose", 1),
        ("starss", "clapping", 1), ("starss", "clapping", 1),
        ("starss", "clapping", 1),
    ]):
        rows.append(dict(id=str(i), source=source, task=task, split="train",
                         observation=f"{source}:recording-{i}:window", group=f"room-{i % 2}",
                         options=["no", "yes"], queries=[task + ":no", task + ":yes"],
                         question=task + "?", target_index=target))
    return rows, [list(range(8))] * 2


def test_repair_is_deterministic_unique_and_keeps_all_other_slots():
    from scripts.native_support_sampling import repair_support

    rows, order = fixture()
    old_rows, old_order = copy.deepcopy(rows), copy.deepcopy(order)
    result = repair_support(rows, order)
    assert result == repair_support(rows, order)
    assert rows == old_rows and order == old_order
    assert result["events"]["clapping"]["after_positive_unique"] == 2
    assert len(result["replacements"]) == 2
    assert len(set(i for b in result["order"] for i in b)) == 8
    for old, new in zip(order, result["order"], strict=True):
        assert old[4:] == new[4:]
    assert sorted(Counter(i for b in result["order"] for i in b).values()) == [2] * 8


def test_capped_support_does_not_duplicate_scarce_positive():
    from scripts.native_support_sampling import repair_support

    rows, order = fixture()
    rows = rows[:9]
    result = repair_support(rows, order)
    event = result["events"]["clapping"]
    assert event["target_positive_unique"] == event["after_positive_unique"] == 1
    assert event["available_positive_unique"] == 1
    assert len(result["replacements"]) == 1


def test_unsupported_class_and_already_supported_event_remain_fixed():
    from scripts.native_support_sampling import repair_support

    rows, order = fixture()
    result = repair_support(rows[:8], order)
    assert result["order"] == order
    assert result["events"]["clapping"]["status"] == "NO_AVAILABLE_TRAIN_POSITIVE"
    assert result["events"]["speech"]["status"] == "ALREADY_MEETS_FLOOR"


@pytest.mark.parametrize("change", ["validation", "duplicate", "wrong_options", "unknown_index"])
def test_invalid_training_inventory_or_order_rejected(change):
    from scripts.native_support_sampling import repair_support

    rows, order = fixture()
    if change == "validation":
        rows[-1]["split"] = "validation"
    elif change == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    elif change == "wrong_options":
        rows[0]["options"] = ["yes", "no"]
    else:
        order[0][0] = len(rows)
    with pytest.raises(ValueError):
        repair_support(rows, order)


def test_candidate_question_or_query_mismatch_cannot_be_substituted():
    from scripts.native_support_sampling import repair_support

    rows, order = fixture()
    for r in rows[8:]:
        r["question"] = "different question"
    with pytest.raises(ValueError, match="condition"):
        repair_support(rows, order)


def test_fixed_config_keeps_architecture_budget_and_initialization():
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "configs/pretrained_reuse"
    old = json.loads((base / "native_long_budget_v1.json").read_text())
    new = json.loads((base / "native_support_v1.json").read_text())
    assert old["training"] == new["training"]
    for key in ("corpus", "queries", "initial_readout", "state_plan"):
        assert old[key] == new[key]
    assert new["training"]["max_steps"] == 32768
    assert new["training"]["performance_early_stopping"] is False


def test_runtime_loss_optimizer_and_selector_are_unchanged():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    trees = [ast.parse((root / name).read_text(encoding="utf-8")) for name in
             ("train_native_long_budget.py", "train_native_support.py")]
    # Compare actual optimizer construction and update loop, not just configuration labels.
    def selected(tree):
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "step"]
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 ((isinstance(n.func, ast.Attribute) and n.func.attr == "AdamW") or
                  (isinstance(n.func, ast.Name) and n.func.id == "choose_best"))]
        return [ast.dump(n, include_attributes=False) for n in loops + calls]

    assert selected(trees[0]) == selected(trees[1])
