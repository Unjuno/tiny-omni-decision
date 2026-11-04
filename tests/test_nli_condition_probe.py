import pytest

from scripts.evaluate_nli_condition_probe import examples, hypothesis


def test_hypothesis_retains_condition_and_candidate_without_numeric_answers():
    rows = examples()
    for row in rows:
        for option in row["options"]:
            text = hypothesis(row, option)
            assert text == f"Candidate {option} has the {row['direction']} {row['field']}."
            assert not any(c.isdigit() for c in text)
        modified = dict(row, target=(row["target"]+1)%3)
        assert hypothesis(row, "A") == hypothesis(modified, "A")
    with pytest.raises(ValueError):
        hypothesis(dict(rows[0], field="target"), "A")
