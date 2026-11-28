# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_physionpp_videoprism_path_a import (
    candidate_features,
    load_feature_cache,
    metric_values,
    validate_config,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def test_candidate_feature_encodes_semantic_option_order_invariantly() -> None:
    video = np.linspace(-1.0, 1.0, 768, dtype=np.float32)
    canonical = candidate_features(video, ["No", "Yes"])
    reversed_order = candidate_features(video, ["Yes", "No"])
    assert canonical.shape == (2, 770)
    assert np.array_equal(canonical[0], reversed_order[1])
    assert np.array_equal(canonical[1], reversed_order[0])
    assert canonical[0, -2:].tolist() == [1.0, 0.0]
    assert canonical[1, -2:].tolist() == [0.0, 1.0]


def test_candidate_features_reject_invalid_video_or_options() -> None:
    with pytest.raises(ValueError, match="768-D"):
        candidate_features(np.zeros(16, dtype=np.float32), ["No", "Yes"])
    with pytest.raises(ValueError, match="options"):
        candidate_features(np.zeros(768, dtype=np.float32), ["Maybe", "Yes"])


def test_shared_candidate_scorer_is_order_equivariant() -> None:
    torch.manual_seed(17)
    scorer = FrozenFeatureCandidateScorer(770, 128).eval()
    features = torch.from_numpy(candidate_features(np.ones(768, dtype=np.float32), ["No", "Yes"]))
    canonical = scorer(features)
    reversed_logits = scorer(features.flip(0)).flip(0)
    assert torch.equal(canonical, reversed_logits)
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 98_817


def test_metric_values_returns_calibrated_binary_metrics() -> None:
    logits = torch.tensor([[4.0, -1.0], [-1.0, 4.0]])
    targets = torch.tensor([0, 1])
    result = metric_values(logits, targets)
    assert result["accuracy"] == 1.0
    assert result["nll"] < 0.02
    assert result["brier"] < 0.001
    assert result["ece_15"] < 0.01
    assert result["count"] == 2


def test_load_feature_cache_validates_hash_ids_order_shape_and_finiteness(tmp_path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text("{}\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    rows = [
        {"id": "x", "media": [{"sha256": "a" * 64}]},
        {"id": "y", "media": [{"sha256": "b" * 64}]},
    ]
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        sample_ids=np.asarray(["x", "y"]),
        media_sha256=np.asarray(["a" * 64, "b" * 64]),
        features=np.ones((2, 768), dtype=np.float32),
    )
    report = {
        "status": "complete_train_feature_cache",
        "model_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
        "checkpoint_sha256": "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29",
        "frame_count": 8,
        "frame_size": 288,
        "feature_dimension": 768,
        "device_policy": "JAX_PLATFORMS=cpu",
        "train_manifest_sha256": manifest_hash,
        "feature_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "feature_ids_in_manifest_order": ["x", "y"],
        "feature_media_sha256_in_manifest_order": ["a" * 64, "b" * 64],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    features = load_feature_cache(
        cache,
        report_path,
        rows,
        expected_split="train",
        expected_manifest_sha256=manifest_hash,
    )
    assert features.shape == (2, 768)
    assert np.isfinite(features).all()

    report["feature_ids_in_manifest_order"] = ["y", "x"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="ID/media order"):
        load_feature_cache(
            cache,
            report_path,
            rows,
            expected_split="train",
            expected_manifest_sha256=manifest_hash,
        )


def test_frozen_experiment_config_matches_registered_candidate() -> None:
    from pathlib import Path

    import yaml

    config_path = (
        Path(__file__).parents[1]
        / "configs/pretrained_reuse/path_a_video_videoprism_physionpp_contact_seed17.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    config["readout"]["epochs"] = 24
    with pytest.raises(ValueError, match="readout/training policy"):
        validate_config(config)
