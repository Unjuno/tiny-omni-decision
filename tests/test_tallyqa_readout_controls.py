from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_tallyqa_vg_readout import (  # noqa: E402
    _batch_logits,
    _cache_features,
    _encode_rows,
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


def test_readout_encoding_supports_sixteen_options() -> None:
    question = torch.zeros(384)
    options = [torch.full((384,), float(index)) for index in range(16)]
    rows = [
        {
            "question_id": 7,
            "image_id": 11,
            "options": [{"id": f"count_{i}"} for i in range(16)],
            "target": 13,
        }
    ]
    text_features = {"7": question}
    text_features.update({f"option:7:{index}": value for index, value in enumerate(options)})
    tasks, targets = _encode_rows(rows, {"11": torch.ones(768)}, text_features)
    assert tasks[0].shape == (16, 1536)
    assert targets == [13]
    assert torch.equal(tasks[0][:, 768:1152], question.expand(16, -1))


def test_readout_fit_and_metrics_support_sixteen_options() -> None:
    tasks = [torch.randn(16, 6) for _ in range(4)]
    targets = [0, 5, 10, 15]
    scorer, fit = _fit(
        tasks,
        targets,
        validation_tasks=tasks[:2],
        validation_targets=targets[:2],
        feature_size=6,
        seed=17,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
    )
    logits = _batch_logits(scorer, tasks[:1], torch.device("cpu"))
    assert logits.shape == (1, 16)
    assert fit["history"][0]["validation"]["count"] == 2


def test_readout_encoding_rejects_inconsistent_option_counts() -> None:
    rows = [
        {"question_id": 1, "image_id": 1, "options": [{}, {}], "target": 0},
        {"question_id": 2, "image_id": 2, "options": [{}, {}, {}], "target": 1},
    ]
    features: dict[str, torch.Tensor] = {"1": torch.zeros(384), "2": torch.zeros(384)}
    for qid, count in ((1, 2), (2, 3)):
        features.update({f"option:{qid}:{i}": torch.zeros(384) for i in range(count)})
    try:
        _encode_rows(rows, {"1": torch.zeros(768), "2": torch.zeros(768)}, features)
    except ValueError as error:
        assert "same option count" in str(error)
    else:
        raise AssertionError("mixed option cardinality must fail before batching")


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


def test_feature_cache_can_serve_grouped_subset_without_cross_split_leak(tmp_path) -> None:
    cache = tmp_path / "features.npz"
    np.savez_compressed(
        cache,
        train_ids=np.asarray(["1", "2", "3"]),
        train_media_sha256=np.asarray(["hash-1", "hash-2", "hash-3"]),
        train_features=np.arange(3 * 768, dtype=np.float32).reshape(3, 768),
        validation_ids=np.asarray(["4"]),
        validation_media_sha256=np.asarray(["hash-4"]),
        validation_features=np.ones((1, 768), dtype=np.float32),
    )
    features = _cache_features(
        cache,
        [{"image_id": 1, "media_sha256": "hash-1"}],
        [{"image_id": 2, "media_sha256": "hash-2"}],
    )
    assert set(features) == {"1", "2"}
