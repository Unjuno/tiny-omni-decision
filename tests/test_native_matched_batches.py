from collections import Counter

import pytest


def corpus():
    return [
        dict(
            id=f"{m}:{i}",
            split="train",
            modality=m,
            source=m,
            task="choose",
            observation=f"{m}:state:{i}",
            question="Choose the value.",
            options=["one", "two"],
            queries=["q1", "q2"],
            target_index=i % 2,
        )
        for m in ("text", "image", "audio", "video")
        for i in range(4)
    ]


def test_both_policies_preserve_all_occurrences_and_balance():
    from scripts.native_matched_batches import build_plan

    rows = corpus()
    pool = list(range(16)) + list(range(16))
    for policy in ("independent", "matched"):
        plan = build_plan(rows, pool, policy)
        assert plan == build_plan(rows, pool, policy)
        assert sorted(o for b in plan["occurrences"] for o in b) == list(range(32))
        assert Counter(pool[o] for b in plan["occurrences"] for o in b) == Counter(pool)
        for batch in plan["occurrences"]:
            assert Counter(rows[pool[o]]["modality"] for o in batch) == {
                "text": 2,
                "image": 2,
                "audio": 2,
                "video": 2,
            }
    plan = build_plan(rows, pool, "matched")
    assert plan["matched_pairs"] == 16
    for batch in plan["occurrences"]:
        for a, b in zip(batch[::2], batch[1::2], strict=True):
            a, b = rows[pool[a]], rows[pool[b]]
            assert a["observation"] != b["observation"]
            assert a["options"][a["target_index"]] != b["options"][b["target_index"]]


def test_unpairable_rows_are_retained_with_explicit_fallback():
    from scripts.native_matched_batches import build_plan

    rows = corpus()
    for row in rows:
        row["target_index"] = 0
    plan = build_plan(rows, list(range(16)), "matched")
    assert plan["matched_pairs"] == 0
    assert plan["fallback_pairs"] == 8
    assert sorted(o for b in plan["occurrences"] for o in b) == list(range(16))


def test_conditions_sources_assets_and_queries_are_checked():
    from scripts.native_matched_batches import build_plan, eligible

    rows = corpus()
    a, b = rows[:2]
    assert eligible(a, b)
    assert not eligible(a, dict(b, source="other"))
    assert not eligible(a, dict(b, observation=a["observation"]))
    assert not eligible(a, dict(b, question="Other condition"))
    assert not eligible(a, dict(b, options=["three", "four"]))
    reordered = dict(b, options=b["options"][::-1], queries=b["queries"][::-1], target_index=0)
    assert eligible(a, reordered)
    rows[1]["queries"] = ["different", "q2"]
    with pytest.raises(ValueError, match="query"):
        build_plan(rows, list(range(16)), "matched")


@pytest.mark.parametrize("problem", ["split", "index", "balance", "policy", "duplicate"])
def test_invalid_plans_fail_closed(problem):
    from scripts.native_matched_batches import build_plan

    rows, pool, policy = corpus(), list(range(16)), "matched"
    if problem == "split":
        rows[0]["split"] = "validation"
    elif problem == "index":
        pool[0] = -1
    elif problem == "balance":
        pool[0] = 4
    elif problem == "policy":
        policy = "cherry_pick"
    else:
        rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        build_plan(rows, pool, policy)


def test_occurrence_permutation_and_fixed_new_schedule():
    from scripts.native_fit import option_order
    from scripts.native_matched_batches import learning_rate

    row = corpus()[0]
    # Occurrence identity, never its reordered update, is the permutation seed.
    assert option_order(row, 5) == option_order(dict(row), 5)
    assert learning_rate(1) == pytest.approx(0.001 / 31)
    assert learning_rate(31) == 0.001
    assert learning_rate(512) > 0
    assert learning_rate(1024) == 0
    with pytest.raises(ValueError):
        learning_rate(1025)
