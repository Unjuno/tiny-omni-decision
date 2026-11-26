from collections import Counter

import pytest


def fixture():
    rows, pool = [], []
    for mod in ("text", "image", "audio", "video"):
        for i in range(2 if mod == "video" else 6):
            if i < 2:
                pool.append(len(rows))
            rows.append(
                dict(
                    id=f"{mod}:{i}",
                    split="train",
                    modality=mod,
                    source=mod,
                    task="choose",
                    observation=f"{mod}:{i}",
                    group=f"{mod}:{i}",
                    options=["a", "b"],
                    target_index=i % 2,
                )
            )
    return rows, pool


def test_fresh_replaces_repeats_with_unused_rows_without_changing_quota_sequence():
    from scripts.native_coverage import make_pool, quota_key

    rows, old = fixture()
    repeat = make_pool(rows, old, "repeat")
    fresh = make_pool(rows, old, "fresh")
    assert repeat["pool"] == old + old
    assert fresh == make_pool(rows, old, "fresh")
    assert fresh["pool"][: len(old)] == old
    assert [quota_key(rows[i]) for i in fresh["pool"]] == [
        quota_key(rows[i]) for i in repeat["pool"]
    ]
    assert fresh["new_unique_by_modality"] == dict(text=2, image=2, audio=2, video=0)
    assert len(set(fresh["pool"])) == 14
    assert Counter(
        i for i in fresh["pool"][len(old) :] if rows[i]["modality"] == "video"
    ) == Counter(i for i in old if rows[i]["modality"] == "video")


def test_full_exhaustion_is_recorded_and_original_occurrences_retained():
    from scripts.native_coverage import make_pool

    rows, _ = fixture()
    pool = list(range(len(rows)))
    fresh = make_pool(rows, pool, "fresh")
    assert fresh["pool"] == pool + pool
    assert fresh["extra_repeated_occurrences"] == len(pool)
    assert not any(fresh["new_unique_by_modality"].values())


def test_candidate_count_is_part_of_quota_and_never_reduced():
    from scripts.native_coverage import make_pool

    rows, pool = fixture()
    for r in rows[2:6]:
        r["options"] = ["a", "b", "c"]
    plan = make_pool(rows, pool, "fresh")
    assert plan["new_unique_by_modality"]["text"] == 0
    assert all(len(rows[i]["options"]) == 2 for i in plan["pool"])


@pytest.mark.parametrize("case", ["split", "duplicate", "index", "policy"])
def test_bad_membership_fails_closed(case):
    from scripts.native_coverage import make_pool

    rows, pool = fixture()
    policy = "fresh"
    if case == "split":
        rows[-1]["split"] = "validation"
    elif case == "duplicate":
        rows.append(rows[0])
    elif case == "index":
        pool[0] = -1
    else:
        policy = "lucky"
    with pytest.raises(ValueError):
        make_pool(rows, pool, policy)
