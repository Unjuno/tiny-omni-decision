import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_reference_predictions_require_exact_routing():
    from scripts.report_occlusion_readout import aligned_reference

    rows = [dict(video_id="a"), dict(video_id="b")]
    targets = [0, 1, 1, 0]
    saved = dict(pairs=rows, targets=targets, probabilities=dict(normal=[[0.5, 0.5]] * 4))
    assert aligned_reference(saved, rows, targets).shape == (2, 2, 2)
    with pytest.raises(ValueError, match="identity"):
        aligned_reference(saved, rows[::-1], targets)
    with pytest.raises(ValueError, match="identity"):
        aligned_reference(saved, rows, targets[::-1][1:] + [1])
