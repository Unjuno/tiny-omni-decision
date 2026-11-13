from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.videomae_runtime import (  # noqa: E402
    ENCODER_PARAMETERS,
    FEATURE_WIDTH,
    READOUT_PARAMETERS,
    TaxonomyConditionedScorer,
    image_tensor,
)


def test_image_tensor_matches_static_16_frame_path_a_contract() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[..., 0] = 255

    # Encode through PIL so the test exercises the same byte-based runtime API.
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    tensor = image_tensor(buffer.getvalue())

    assert tuple(tensor.shape) == (3, 16, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor[:, 0], tensor[:, -1])
    assert float(tensor[0].min()) == pytest.approx(1.0)
    assert float(tensor[1].max()) == pytest.approx(-1.0)


def test_scorer_matches_frozen_path_a_parameter_and_shape_contract() -> None:
    model = TaxonomyConditionedScorer()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == READOUT_PARAMETERS == 52_682
    with torch.inference_mode():
        logits = model(torch.zeros((2, FEATURE_WIDTH)), torch.tensor([0, 2]))
    assert tuple(logits.shape) == (2, 10)
    assert sum((ENCODER_PARAMETERS, parameter_count)) == 22_205_968
