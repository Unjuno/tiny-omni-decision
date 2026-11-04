from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="optional Path A model-evaluation dependency")
pytest.importorskip("transformers", reason="optional Path A model-evaluation dependency")

from tiny_omni_decision.text_runtime import (  # noqa: E402
    TypedTextDecisionRuntime,
    candidate_features,
)


def test_candidate_features_preserve_supplied_option_count_and_shape() -> None:
    context = torch.zeros(384)
    options = torch.arange(3 * 384, dtype=torch.float32).reshape(3, 384)

    result = candidate_features(context, options)

    assert result.shape == (3, 1536)
    assert torch.isfinite(result).all()
    assert torch.equal(result[:, :384], context.expand(3, -1))
    assert torch.equal(result[:, 384:768], options)


@pytest.mark.parametrize(
    ("context", "options"),
    [
        (torch.zeros(383), torch.zeros(2, 384)),
        (torch.zeros(384), torch.zeros(1, 384)),
        (torch.zeros(384), torch.zeros(2, 383)),
        (torch.full((384,), float("nan")), torch.zeros(2, 384)),
    ],
)
def test_candidate_features_reject_invalid_inputs(
    context: torch.Tensor, options: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        candidate_features(context, options)


def test_text_runtime_encodes_a_single_nonempty_criterion(monkeypatch) -> None:
    runtime = object.__new__(TypedTextDecisionRuntime)
    encoded_texts: list[str] = []

    def encode(texts):
        encoded_texts.extend(texts)
        return torch.ones((len(texts), 384), dtype=torch.float32)

    monkeypatch.setattr(runtime, "_encode", encode)

    result = runtime.encode_criterion("prefer the higher score")

    assert encoded_texts == ["prefer the higher score"]
    assert result.shape == (384,)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("criterion", ["", "  ", None, 3])
def test_text_runtime_rejects_empty_or_nontext_criterion(criterion) -> None:
    runtime = object.__new__(TypedTextDecisionRuntime)

    with pytest.raises(ValueError, match="nonempty string"):
        runtime.encode_criterion(criterion)
