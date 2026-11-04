import pytest

torch = pytest.importorskip("torch")
from scripts.train_grounding_tokens import score_batch  # noqa: E402
from tiny_omni_decision.token_readout import VideoTokenReadout  # noqa: E402


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_variable_options_match_separate_and_permutation(mode):
    torch.manual_seed(17)
    model = VideoTokenReadout(mode).eval()
    tokens = torch.randn(2, 2048, 768)
    options = [torch.randn(2, 1536), torch.randn(6, 1536)]
    actual = score_batch(model, tokens, options)
    assert [len(v) for v in actual] == [2, 6]
    for i, expected in enumerate(actual):
        single = score_batch(model, tokens[i : i + 1], [options[i]])[0]
        torch.testing.assert_close(expected, single)
    reversed_scores = score_batch(model, tokens, [o.flip(0) for o in options])
    for a, b in zip(actual, reversed_scores, strict=True):
        torch.testing.assert_close(a, b.flip(0))
