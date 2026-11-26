from scripts.occlusion_prior import evaluate


def row(options, a, b):
    return dict(
        options=options,
        questions=[dict(condition="initial", target=a), dict(condition="final", target=b)],
    )


def test_condition_specific_counts_and_unseen_uniform():
    train = [row(["a", "b", "c"], 1, 2)]
    valid = [row(["a", "b", "c"], 1, 2), row(["d", "e", "f"], 0, 0)]
    result = evaluate(train, valid)
    assert result["both_correct"] == 1
    assert result["initial"]["unseen_option_sets"] == 1
    assert result["final"]["accuracy"] == 1
    assert train[0]["questions"][0]["target"] == 1
