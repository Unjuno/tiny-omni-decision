import json

from scripts.evaluate_text_condition_probe import examples


def test_condition_probe_oracle_and_pairing():
    rows = examples()
    assert rows == examples()
    assert len(rows) == len({r["id"] for r in rows}) == 256
    for row in rows:
        values = [s[row["field"]] for s in json.loads(row["state"])]
        assert len(set(values)) == 3
        ordered = sorted(range(3), key=values.__getitem__)
        assert row["target"] == ordered[0 if row["direction"] == "lowest" else -1]
    for index in range(0, 256, 2):
        a, b = rows[index:index+2]
        assert a["state"] == b["state"] and a["options"] == b["options"]
        assert a["target"] != b["target"]


def test_candidate_description_keeps_identifier_and_adds_no_facts():
    import pytest

    from scripts.evaluate_text_condition_probe import candidate_text

    for key in ("A", "B", "C"):
        assert candidate_text(key, "bare") == key
        assert candidate_text(key, "described") == f"{key}: Candidate {key}."
    with pytest.raises(ValueError):
        candidate_text("A", "unknown")
