import pytest

torch = pytest.importorskip("torch")
from scripts.train_scene_trajectory_readout import batch  # noqa: E402


def test_controls_replace_only_intended_block():
    bank = {"a": torch.ones(256, 896), "b": torch.full((256, 896), 2.0)}
    data = dict(
        pairs=[dict(video_id="a"), dict(video_id="b")],
        text=torch.zeros(4, 3, 1536),
        targets=torch.zeros(4, dtype=torch.long),
    )
    for control, geometry, scene in [
        ("normal", 1, 1),
        ("scene_mismatch", 1, 2),
        ("trajectory_mismatch", 2, 1),
        ("mismatch", 2, 2),
        ("zero_scene", 1, 0),
        ("zero_trajectory", 0, 1),
        ("zero", 0, 0),
    ]:
        values, _, _, _ = batch(bank, data, [0], control)
        assert (values[:, :, :512] == geometry).all()
        assert (values[:, :, 512:] == scene).all()
    assert (bank["a"] == 1).all()
