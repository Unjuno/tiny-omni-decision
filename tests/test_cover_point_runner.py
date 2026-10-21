import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.cover_point_metrics import arm_input, membership_metrics  # noqa: E402


def test_initial_control_retains_first_observation_and_appearance_without_mutation():
    points = torch.arange(512.0).reshape(1, 1, 512)
    normal = arm_input(points, "trajectory")
    initial = arm_input(points, "initial_only")
    torch.testing.assert_close(normal, points)
    torch.testing.assert_close(initial[..., :4], points[..., :4])
    torch.testing.assert_close(initial[..., 128:], points[..., 128:])
    assert not initial[..., 4:128].any()
    assert points[..., 4:128].any()
    with pytest.raises(ValueError):
        arm_input(points, "unknown")


def test_metrics_retain_all_negative_clip_and_define_binary_brier():
    p = np.array([[0.9, 0.1], [0.2, 0.3]])
    y = np.array([[True, False], [False, False]])
    m = membership_metrics(p, y)
    assert m["clips"] == 2 and m["points"] == 4
    assert m["grid_supported_clips"] == 1
    assert m["top1_point_in_cover_bbox"] == 0.5
    assert m["natural_grid_accuracy"] == 1
    assert m["brier"] == pytest.approx(2 * np.square(p - y).mean())
    assert m["positive_recall"] == 1 and m["negative_recall"] == 1
    assert m["positive_fraction"] == 0.25
    with pytest.raises(ValueError):
        membership_metrics(np.array([[float("nan")]]), np.array([[False]]))
