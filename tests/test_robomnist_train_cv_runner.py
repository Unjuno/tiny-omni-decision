"""Ensure the real-run entry point cannot expand its TRAIN feature whitelist."""
import json

import pytest

pytest.importorskip("torch")
from scripts import fit_robomnist_train_cv as cv  # noqa: E402


def fixture_manifest(tmp_path):
    train = []
    for pair in range(390):
        groups = [f"g{pair}a", f"g{pair}b"]
        for audio in groups:
            for video in groups:
                train.append({"audio_group": audio, "video_group": video,
                              "pair_unit_id": str(pair), "condition": str(pair % 60),
                              "target": int(audio == video)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"examples": {"train": train, "validation": [
        {"audio_group": "forbidden_audio", "video_group": "forbidden_video"}]}}))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "run.json").write_text(json.dumps({"status": "COMPLETE",
        "sealed_audit_opened": False,
        "effective": {"manifest_sha256": cv.original.MANIFEST_SHA256}}))
    return manifest, cache


def test_loader_receives_only_original_train_id_whitelist(tmp_path, monkeypatch):
    manifest, cache = fixture_manifest(tmp_path)
    monkeypatch.setattr(cv.original, "sha256_file", lambda _: cv.original.MANIFEST_SHA256)
    observed = []

    def loader(root, ids):
        assert root == cache
        assert len(ids) == 780
        assert not {"forbidden_audio", "forbidden_video"} & ids
        observed.append(ids)
        return dict.fromkeys(ids), dict.fromkeys(ids, "verified_hash")

    monkeypatch.setattr(cv.original, "_load_group_features", loader)
    rows, folds, features, _, _ = cv.load_train(cache, manifest)
    assert len(rows) == 1560 and len(folds) == 390 and len(features) == 780
    assert len(observed) == 1


def test_overlap_fails_before_any_feature_open(tmp_path, monkeypatch):
    manifest, cache = fixture_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["examples"]["validation"][0]["audio_group"] = "g0a"
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(cv.original, "sha256_file", lambda _: cv.original.MANIFEST_SHA256)

    def must_not_read(*args):
        pytest.fail("features were opened before split validation")

    monkeypatch.setattr(cv.original, "_load_group_features", must_not_read)
    with pytest.raises(ValueError, match="identity/count"):
        cv.load_train(cache, manifest)


def test_scale_arm_changes_only_cosine_multiplier():
    import math
    import torch

    torch.manual_seed(17)
    baseline = cv.BilinearAVMatchScorer(rank=32)
    unchanged = cv.ScoreScaleScorer(math.sqrt(32))
    unit = cv.ScoreScaleScorer(1.0)
    unchanged.load_state_dict(baseline.state_dict())
    unit.load_state_dict(baseline.state_dict())
    audio, video = torch.randn(4, 768), torch.randn(4, 1536)
    assert torch.equal(baseline(audio, video), unchanged(audio, video))
    torch.testing.assert_close(unit(audio, video) * math.sqrt(32), baseline(audio, video))
    assert unit.trainable_parameters == baseline.trainable_parameters == 73728
