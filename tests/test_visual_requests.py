import pytest

from tiny_omni_decision.visual_requests import VisualRequest


def test_profile_shapes_and_bounds():
    asset = dict(path="local.png", sha256="a" * 64)
    assert VisualRequest(key="image", modality="image", assets=[asset] * 8).output_shape() == (
        8, 17, 768)
    assert VisualRequest(key="video", modality="video", assets=[asset]).output_shape() == (
        1, 2048, 768)
    for args in (dict(modality="video", assets=[asset] * 2),
                 dict(modality="image", assets=[asset] * 9),
                 dict(modality="audio", assets=[asset]),
                 dict(modality="image", assets=[dict(asset, sha256="bad")])):
        with pytest.raises(ValueError):
            VisualRequest(key="x", **args)
