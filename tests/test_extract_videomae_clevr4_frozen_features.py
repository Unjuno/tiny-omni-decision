from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from scripts.extract_videomae_clevr4_frozen_features import image_tensor  # noqa: E402
from scripts.train_videomae_clevr4_option_scorer import (  # noqa: E402
    TaxonomyConditionedScorer,
    decision_json,
    parameter_count,
    score_options,
)


def test_static_image_is_repeated_and_normalized_for_all_frames() -> None:
    pixels = np.zeros((12, 20, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")

    result = image_tensor(buffer.getvalue())

    assert tuple(result.shape) == (3, 16, 224, 224)
    assert result.dtype == torch.float32
    assert result[0, :, 0, 0].tolist() == [1.0] * 16
    assert result[1, :, 0, 0].tolist() == [-1.0] * 16
    assert result[2, :, 0, 0].tolist() == [-1.0] * 16
    assert result.is_contiguous()


def test_option_scorer_preserves_caller_order_and_scores_subsets() -> None:
    torch.manual_seed(17)
    model = TaxonomyConditionedScorer()
    feature = torch.randn(384)
    first_order = ["gray", "red", "blue"]
    second_order = ["blue", "gray", "red"]

    names_a, probabilities_a = score_options(
        model, feature, "color", first_order, torch.device("cpu")
    )
    names_b, probabilities_b = score_options(
        model, feature, "color", second_order, torch.device("cpu")
    )

    assert names_a == first_order
    assert names_b == second_order
    mapped_a = dict(zip(names_a, probabilities_a, strict=True))
    mapped_b = dict(zip(names_b, probabilities_b, strict=True))
    assert mapped_a.keys() == mapped_b.keys()
    for option in mapped_a:
        assert abs(mapped_a[option] - mapped_b[option]) < 1e-7
    assert abs(sum(probabilities_a) - 1.0) < 1e-7
    assert parameter_count(model) == 52_682

    record = json.loads(
        decision_json(model, feature, "color", first_order, "sample-1", torch.device("cpu"))
    )
    assert record["request_id"] == "clevr4.color:sample-1"
    assert [item["id"] for item in record["options"]] == first_order
    assert abs(sum(item["probability"] for item in record["options"]) - 1.0) < 1e-6


def test_option_scorer_rejects_unknown_or_duplicate_options() -> None:
    model = TaxonomyConditionedScorer()
    feature = torch.zeros(384)
    with pytest.raises(ValueError, match="not valid"):
        score_options(model, feature, "color", ["gray", "not-a-color"], torch.device("cpu"))
    with pytest.raises(ValueError, match="unique"):
        score_options(model, feature, "color", ["gray", "gray"], torch.device("cpu"))
