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


from scripts.report_video_grounding import compare  # noqa: E402


def test_grounding_scene_bootstrap_identity_and_mismatch():
    rows = [
        dict(id=str(i), scene_index=i, taxonomy="count", target_index=0, probabilities=[0.7, 0.3])
        for i in range(3)
    ]
    result = compare(rows, rows)
    assert result["accuracy"]["difference"] == 0
    assert result["nll"]["percentile_95_interval"] == [0, 0]
    with pytest.raises(ValueError, match="unpaired"):
        compare(rows, [{**r, "id": "changed"} for r in rows])
