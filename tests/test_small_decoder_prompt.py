import pytest

from tiny_omni_decision.small_decoder_prompt import decision_prompt, option_labels


def test_target_and_metadata_never_enter_prompt():
    row = dict(
        state='[{"id":"K","cost":1}]',
        question="Which item?",
        options=["K", "Z"],
        target="K",
        oracle="secret",
        split="train",
    )
    before = decision_prompt(row)
    row["target"] = "Z"
    row["oracle"] = "changed"
    row["split"] = "validation"
    assert before == decision_prompt(row)
    assert "A: K\nB: Z" in before
    assert "secret" not in before


def test_all_options_preserved_without_target_hint():
    row = dict(state="s", question="q", options=["x", "y", "z"])
    assert "C: z" in decision_prompt(row)
    assert option_labels(3) == ["A", "B", "C"]


@pytest.mark.parametrize("n", [0, 1, 27])
def test_invalid_option_counts(n):
    with pytest.raises(ValueError):
        option_labels(n)
