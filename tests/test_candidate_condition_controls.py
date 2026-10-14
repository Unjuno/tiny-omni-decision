import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.candidate_condition_controls import question_control  # noqa: E402


def candidates():
    torch.manual_seed(17)
    q = torch.randn(2, 2, 1, 384).expand(-1, -1, 3, -1)
    o = torch.randn(2, 1, 3, 384).expand(-1, 2, -1, -1)
    return torch.cat((q, o, (q - o).abs(), q * o), -1)


def test_controls_rebuild_only_question_preserve_options_and_source():
    values = candidates()
    original = values.clone()
    torch.testing.assert_close(question_control(values, "normal"), values, rtol=0, atol=0)
    swapped = question_control(values, "swapped_question")
    torch.testing.assert_close(swapped, values.flip(1), rtol=0, atol=0)
    neutral = question_control(values, "neutral_question")
    torch.testing.assert_close(neutral[:, 0], neutral[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(neutral[..., 384:768], values[..., 384:768], rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_paired_encoder_rounding_preserved_without_replacing_options():
    values = candidates()
    q, o = values[..., :384], values[..., 384:768].clone()
    o[0, 1, 0, 0] = torch.nextafter(o[0, 1, 0, 0], torch.tensor(torch.inf))
    values = torch.cat((q, o, (q - o).abs(), q * o), -1)
    for mode in ("normal", "neutral_question", "swapped_question"):
        result = question_control(values, mode)
        torch.testing.assert_close(result[..., 384:768], o, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["factor", "options", "questions", "empty", "nan", "mode"])
def test_invalid_factorization_or_condition_pair_fails_closed(kind):
    values = candidates()
    mode = "normal"
    if kind == "factor":
        values[0, 0, 0, 800] += 1
    elif kind == "options":
        q, o = values[..., :384], values[..., 384:768].clone()
        o[0, 1, 0] += 1
        values = torch.cat((q, o, (q - o).abs(), q * o), -1)
    elif kind == "questions":
        q, o = values[..., :384].clone(), values[..., 384:768]
        q[0, 0, 0] += 1
        values = torch.cat((q, o, (q - o).abs(), q * o), -1)
    elif kind == "empty":
        values = values[:0]
    elif kind == "nan":
        values[0, 0, 0, 0] = float("nan")
    else:
        mode = "unknown"
    with pytest.raises(ValueError):
        question_control(values, mode)
