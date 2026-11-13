from collections import Counter

from scripts.paired_wording_schedule import schedule


def test_matched_parent_exposure_and_view_rotation():
    original = [dict(id=f"o{i}", source="original") for i in range(4096)]
    parents = [dict(id=f"c{i}", source="curriculum", group=str(i // 8),
                    task="extremum" if i % 8 < 6 else "count",
                    state="state", options=["a", "b"], target="a") for i in range(1024)]
    views = [dict(p, id=f"{p['id']}-{v}", parent_id=p["id"], view=v)
             for p in parents for v in (range(4) if p["task"] == "extremum" else [0])]
    args = (original + parents, views, [r["id"] for r in original])
    control = schedule(*args, "control")
    treatment = schedule(*args, "wording")
    assert treatment == schedule(*args, "wording")
    def ids(rows):
        return [r.get("parent_id", r["id"]) for r in rows]
    assert ids(control) == ids(treatment)
    assert len(control) == 8192
    assert len(set(ids(control))) == 5120
    assert Counter(r["source"] for r in treatment) == {"original": 4096, "curriculum": 4096}
    for i in range(0, 8192, 8):
        assert [r["source"] for r in treatment[i:i+8]] == ["original"] * 4 + ["curriculum"] * 4
    by_parent = {}
    for r in treatment:
        if r.get("task") == "extremum":
            by_parent.setdefault(r["parent_id"], []).append(r["view"])
    assert all(v == [0, 1, 2, 3] for v in by_parent.values())
