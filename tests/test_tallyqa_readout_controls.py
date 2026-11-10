from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_tallyqa_vg_readout import (  # noqa: E402
    _batch_logits,
    _cache_features,
    _fit,
)


def test_readout_parameter_counts_match_declared_towers() -> None:
    joint_tasks = [torch.randn(4, 1536) for _ in range(8)]
    question_tasks = [torch.randn(4, 768) for _ in range(8)]
    targets = [0, 1, 2, 3, 0, 1, 2, 3]
    scorer, joint = _fit(
        joint_tasks,
        targets,
        validation_tasks=joint_tasks[:4],
        validation_targets=targets[:4],
        feature_size=1536,
        seed=17,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
    )
    _, question = _fit(
        question_tasks,
        targets,
        validation_tasks=question_tasks[:4],
        validation_targets=targets[:4],
        feature_size=768,
        seed=17,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
    )
    assert joint["trainable_parameters"] == 196_865
    assert question["trainable_parameters"] == 98_561
    assert joint["validation_examples_per_epoch"] == 4
    assert joint["history"][0]["train_metrics"]["count"] == 8
    assert joint["history"][0]["validation"]["count"] == 4
    assert joint["checkpoint_selection"] == "minimum_validation_nll"
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 196_865


def test_image_zero_control_changes_only_visual_feature_slice() -> None:
    tasks = [torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)]
    model = torch.nn.Linear(5, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    original = _batch_logits(model, tasks, torch.device("cpu"))
    zeroed = _batch_logits(model, tasks, torch.device("cpu"), image_zero=True)
    assert not torch.equal(original, zeroed)
    assert torch.equal(tasks[0], torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5))


def test_shuffled_image_control_uses_one_image_mapping_per_question() -> None:
    tasks = [torch.tensor([[1.0, 10.0], [1.0, 20.0]]), torch.tensor([[2.0, 30.0], [2.0, 40.0]])]
    rows = [{"image_id": 1}, {"image_id": 2}]
    images = {"1": torch.tensor([7.0]), "2": torch.tensor([8.0])}
    scorer = torch.nn.Identity()
    outputs = _batch_logits(
        scorer,
        tasks,
        torch.device("cpu"),
        shuffled_images={"1": images["2"], "2": images["1"]},
        rows=rows,
        image_features=images,
    )
    assert torch.equal(outputs[0, :, 0], torch.tensor([8.0, 8.0]))
    assert torch.equal(outputs[1, :, 0], torch.tensor([7.0, 7.0]))


def test_feature_cache_must_match_manifest_image_hash_pairs(tmp_path) -> None:
    cache = tmp_path / "features.npz"
    np.savez_compressed(
        cache,
        train_ids=np.asarray(["1"]),
        train_media_sha256=np.asarray(["train-hash"]),
        train_features=np.zeros((1, 768), dtype=np.float32),
        validation_ids=np.asarray(["2"]),
        validation_media_sha256=np.asarray(["validation-hash"]),
        validation_features=np.ones((1, 768), dtype=np.float32),
    )
    features = _cache_features(
        cache,
        [{"image_id": 1, "media_sha256": "train-hash"}],
        [{"image_id": 2, "media_sha256": "validation-hash"}],
    )
    assert set(features) == {"1", "2"}
    try:
        _cache_features(
            cache,
            [{"image_id": 1, "media_sha256": "wrong-hash"}],
            [{"image_id": 2, "media_sha256": "validation-hash"}],
        )
    except ValueError as error:
        assert "media hash" in str(error)
    else:
        raise AssertionError("a cache whose image/hash pair differs must be rejected")
