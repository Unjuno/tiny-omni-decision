import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_conditioned_features_keep_all_options_and_correct_question_row():
    from scripts.train_vilt_image_probe import _tasks
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    row = {"id": "image", **{k: v[-1] for k, v in CLEVR4_TAXONOMIES.items()}}
    features = {"image": torch.arange(4.0)[:, None].expand(4, 768)}
    options = {
        f"{k}|{v}": torch.zeros(384) for k, values in CLEVR4_TAXONOMIES.items() for v in values
    }
    tasks, targets, groups, kinds = _tasks([row], features, options)
    assert targets == [9] * 4 and groups == ["image"] * 4
    assert kinds == list(CLEVR4_TAXONOMIES)
    for i, task in enumerate(tasks):
        assert task.shape == (10, 1152)
        assert torch.all(task[:, :768] == i)
