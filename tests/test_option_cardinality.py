import pytest

from scripts.analyze_option_cardinality import summarize


def row(p, target="a"):
    return dict(id="one", options=list("abc")[:len(p)], probabilities=p, target=target)


def test_all_distractors_included_and_no_false_full_success():
    result = summarize([row([.3, .5, .2])])["3"]
    assert result["accuracy"] == 0
    assert result["strict_all_wins"] == 0
    assert result["pair_win_rate_ties_half"] == .5
    assert result["gold_distractor_pairs"] == 2


def test_ties_and_binary_identity():
    result = summarize([row([.5, .5])])["2"]
    assert result["pair_win_rate_ties_half"] == .5
    assert result["strict_all_wins"] == 0
    assert result["original_nll"] == result["pair_nll"]


def test_reject_duplicate_and_invalid_probabilities():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row([.6, .4]), row([.6, .4])])
    with pytest.raises(ValueError, match="probabilities"):
        summarize([row([.6, float("nan")])])
