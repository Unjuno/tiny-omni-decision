import json

import pytest

torch = pytest.importorskip("torch")

from scripts.train_joint_video_readout import fit, load_features  # noqa: E402


def test_fixed_fit_selection_and_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    train = torch.randn(4, 3, 192)
    valid = torch.randn(4, 3, 192)
    targets = torch.tensor([0, 1, 1, 2])
    result = fit(train, valid, targets, targets, torch.zeros(192), tmp_path / "run")
    history = json.loads((tmp_path / "run/history.json").read_text())
    assert len(history) == 51
    assert result["best_epoch"] == min(history, key=lambda x: x["valid"]["nll"])["epoch"]
    assert result["exact_reload"]
    assert result["trainable_parameters"] == 192
    assert (tmp_path / "run/final.pt").exists()


def test_partial_features_rejected(tmp_path):
    (tmp_path / "predictions.jsonl").write_text("")
    (tmp_path / "completion.json").write_text('{"rows":0}')
    with pytest.raises(ValueError, match="complete"):
        load_features(tmp_path, [dict(video_id="v")])
