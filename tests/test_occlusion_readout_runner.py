import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_batch_controls_keep_questions_and_targets():
    from scripts.train_occlusion_readout import batch

    bank = {"a": torch.arange(3.0).view(3, 1).expand(3, 768), "b": torch.full((2, 768), 8.0)}
    data = dict(
        pairs=[dict(video_id="a"), dict(video_id="b")],
        text=torch.randn(4, 3, 1536),
        targets=torch.tensor([0, 1, 1, 2]),
    )
    normal = batch(bank, data, [0, 1])
    for control in ("zero", "mismatch", "reverse"):
        modified = batch(bank, data, [0, 1], control)
        torch.testing.assert_close(normal[1], modified[1])
        torch.testing.assert_close(normal[3], modified[3])
    assert batch(bank, data, [0], "reverse")[0][0, :, 0].tolist() == [2, 1, 0]
    assert batch(bank, data, [0], "mismatch")[0][0, :, 0].tolist() == [8, 8]
    assert batch(bank, data, [0], "zero")[0].count_nonzero() == 0


def test_paired_summary_and_strata():
    from scripts.train_occlusion_readout import summarize

    targets = np.array([0, 1, 2, 2])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    result = summarize(probabilities, targets)
    assert result["accuracy"] == 0.75
    assert result["initial"]["accuracy"] == 1
    assert result["final"]["accuracy"] == 0.5
    assert result["both_correct"] == 0.5
    assert result["changed"]["both_correct"] == 1
    assert result["unchanged"]["both_correct"] == 0


def test_incomplete_extraction_fails_closed(tmp_path, monkeypatch):
    import json

    import scripts.train_occlusion_readout as runner

    pairs = tmp_path / "perception-occlusion-pairs-01"
    pairs.mkdir()
    pins = {}
    for split in ("train", "valid"):
        path = pairs / f"{split}.json"
        path.write_text(json.dumps(dict(pairs=[])))
        pins[split] = runner.sha(path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PINS", pins)
    video, text = tmp_path / "video", tmp_path / "text"
    video.mkdir()
    text.mkdir()
    (video / "result.json").write_text('{"status":"RUNNING"}')
    (text / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="complete video"):
        runner.load_inputs(video, text)
