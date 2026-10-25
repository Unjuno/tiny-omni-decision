from scripts.train_openimages_text_only_control import _linear_scorer_parameters


def test_linear_scorer_parameter_accounting_matches_candidate_scorers():
    assert _linear_scorer_parameters(384, 128) == 49_409
    assert _linear_scorer_parameters(768 + 384, 128) == 147_713
