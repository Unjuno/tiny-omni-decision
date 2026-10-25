import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.extract_synthetic_av_readout_features import merge_feature_caches  # noqa: E402
from scripts.train_synthetic_av_readouts import load_training_data  # noqa: E402
from tiny_omni_decision.synthetic_av_task import build_manifest  # noqa: E402
from tiny_omni_decision.synthetic_av_training import (  # noqa: E402
    BestByNLL,
    classification_metrics,
    epoch_permutations,
)


def test_classification_metrics_report_binary_accuracy_nll_brier_ece():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    targets = torch.tensor([0, 1])
    result = classification_metrics(logits, targets, ece_bins=10)
    assert result["accuracy"] == 0.5
    assert result["nll"] > 0.5
    assert 0.0 <= result["brier"] <= 2.0
    assert 0.0 <= result["ece"] <= 1.0
    assert result["mean_confidence"] > 0.5


def test_accuracy_uses_logits_argmax_when_float32_softmax_rounds_to_a_tie():
    logits = torch.tensor([[0.0, 1e-8]], dtype=torch.float32)
    assert torch.equal(torch.softmax(logits, dim=-1), torch.tensor([[0.5, 0.5]]))
    result = classification_metrics(logits, torch.tensor([1]))
    assert result["accuracy"] == 1.0


def test_checkpoint_selector_uses_lowest_nll_and_keeps_earliest_tie():
    selector = BestByNLL()
    assert selector.update(0.8, 512)
    assert not selector.update(0.8, 1024)
    assert selector.step == 512
    assert selector.update(0.7, 1536)
    assert selector.step == 1536


def test_epoch_permutations_are_deterministic_and_cover_each_sample_once():
    first = epoch_permutations(64, epochs=5, seed=17)
    assert first == epoch_permutations(64, epochs=5, seed=17)
    assert all(sorted(epoch) == list(range(64)) for epoch in first)


def _write_fake_feature_cache(root, manifest, groups, smoke_subset):
    root.mkdir()
    render_dir = root / "render"
    feature_dir = root / "features"
    render_dir.mkdir()
    feature_dir.mkdir()
    manifest_path = render_dir / "render-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selected_groups = [group for group in manifest["groups"] if group["group_id"] in groups]
    sequence_ids = list(
        dict.fromkeys(
            sequence
            for group in selected_groups
            for sequence in group["sequence_ids"].values()
        )
    )
    merged = merge_feature_caches(
        manifest,
        sequence_ids,
        np.zeros((len(sequence_ids), 8, 768), dtype=np.float32),
        [0, 1, 2, 3],
        np.zeros((4, 384), dtype=np.float32),
    )
    cache_path = feature_dir / "examples.npz"
    np.savez_compressed(cache_path, **merged)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "status": "PASS_SYNTHETIC_AV_ALIGNED_FEATURE_CACHE",
        "input_manifest_sha256": digest(manifest_path),
        "output_sha256": digest(cache_path),
        "selected_group_ids": groups,
        "smoke_subset": smoke_subset,
    }
    (feature_dir / "feature-report.json").write_text(json.dumps(report), encoding="utf-8")
    return render_dir, feature_dir


def test_training_data_loader_accepts_complete_splits_and_gated_group_smoke(tmp_path):
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    all_groups = [group["group_id"] for group in manifest["groups"]]
    render_dir, feature_dir = _write_fake_feature_cache(
        tmp_path / "full", manifest, all_groups, smoke_subset=False
    )
    loaded = load_training_data(render_dir, feature_dir)
    assert len(loaded["train_ids"]) == 16
    assert len(loaded["validation_ids"]) == 16

    subset = [all_groups[0], all_groups[-1]]
    render_dir, feature_dir = _write_fake_feature_cache(
        tmp_path / "smoke", manifest, subset, smoke_subset=True
    )
    with pytest.raises(ValueError, match="smoke-subset"):
        load_training_data(render_dir, feature_dir)
    loaded = load_training_data(render_dir, feature_dir, allow_smoke=True)
    assert len(loaded["train_ids"]) == 8
    assert len(loaded["validation_ids"]) == 8
    assert set(loaded["train_group_ids"]) == {all_groups[0]}
    assert set(loaded["validation_group_ids"]) == {all_groups[-1]}
