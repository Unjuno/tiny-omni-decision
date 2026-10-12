import pytest

pytest.importorskip("torch")
import torch

from scripts.train_vidimu_fusion_arm import VidimuFusion


def test_wrapper_scores_thirteen_classes_and_masks_missing_video():
    torch.manual_seed(17)
    model = VidimuFusion()
    x = torch.randn(3, 6506)
    p = model(x).softmax(-1)
    assert p.shape == (3, 13) and torch.isfinite(p).all()
    missing = model(x, missing="video")
    x[:, :6144] = float("nan")
    torch.testing.assert_close(missing, model(x, missing="video"), rtol=0, atol=0)
    with pytest.raises(ValueError, match="missing"):
        model(x, missing="unknown")
