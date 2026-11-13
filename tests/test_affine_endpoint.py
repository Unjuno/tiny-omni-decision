import numpy as np
import pytest


def test_endpoint_probabilities_preserve_taxonomy_and_image_order():
    pytest.importorskip("torch")
    from scripts.evaluate_shared_affine_endpoint import prediction_rows
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    labels = [dict(id=str(i), **{k: v[0] for k, v in CLEVR4_TAXONOMIES.items()}) for i in range(2)]
    p = np.full((2, 4, 10), 0.1)
    rows = prediction_rows(labels, p)
    assert [r["id"] for r in rows] == [f"{i}:{k}" for i in range(2) for k in CLEVR4_TAXONOMIES]
    assert all(r["target"] == r["options"][0] for r in rows)
    with pytest.raises(ValueError):
        prediction_rows(labels, p * 2)
    with pytest.raises(ValueError):
        prediction_rows(labels, p[:, :3])
