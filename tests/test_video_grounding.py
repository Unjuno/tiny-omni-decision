import pytest

torch = pytest.importorskip("torch")
from scripts.train_video_grounding import metrics  # noqa: E402


def test_selector_weights_taxonomies_equally():
    rows = [{"taxonomy": "a", "question_type": "static"}] * 3 + [
        {"taxonomy": "b", "question_type": "temporal"}
    ]
    values = [torch.tensor([2.0, 0.0])] * 3 + [torch.tensor([0.0, 2.0])]
    result = metrics(rows, values, [0, 0, 0, 0])
    assert result["overall"]["accuracy"] == 0.75
    assert result["macro_taxonomy"]["accuracy"] == 0.5
    assert result["macro_taxonomy"]["nll"] == pytest.approx(
        (result["taxonomy:a"]["nll"] + result["taxonomy:b"]["nll"]) / 2
    )
