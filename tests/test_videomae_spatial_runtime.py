from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tiny_omni_decision import videomae_spatial_runtime as runtime_module  # noqa: E402
from tiny_omni_decision.videomae_spatial import (  # noqa: E402
    FEATURE_WIDTH,
    SpatialGridClevr4Scorer,
)


def _runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "_load_encoder", lambda *_: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        runtime_module,
        "videomae_spatial_features",
        lambda *_: torch.zeros(1, FEATURE_WIDTH),
    )
    scorer = SpatialGridClevr4Scorer()
    with torch.no_grad():
        first = scorer.network[0]
        last = scorer.network[2]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.copy_(torch.arange(last.bias.numel(), dtype=last.bias.dtype))
    readout = tmp_path / "readout.pt"
    torch.save(scorer.state_dict(), readout)
    digest = hashlib.sha256(readout.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime_module, "SPATIAL_READOUT_SHA256", digest)
    model_root = tmp_path / "model"
    model_root.mkdir()
    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    return (
        runtime_module.VideoMAESpatialGridClevr4Runtime(
            model_root, readout, dependency_root, device="cpu"
        ),
        readout,
        scorer,
    )


def test_spatial_runtime_maps_logits_to_supplied_option_order(tmp_path, monkeypatch) -> None:
    runtime, _, _ = _runtime(tmp_path, monkeypatch)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(image_path)
    media_sha = runtime_module.sha256_file(image_path)

    result = runtime.decide(
        image_path,
        "color",
        ["blue", "red", "gray"],
        expected_media_sha256=media_sha,
    )

    expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0).tolist()
    assert result["probabilities"] == pytest.approx(expected)
    assert result["prediction_index"] == 0
    assert result["confidence"] == pytest.approx(expected[0])
    assert result["feature_shape"] == [1, FEATURE_WIDTH]
    assert result["media_sha256"] == media_sha
    assert not runtime.encoder.weight.requires_grad
    assert not runtime.scorer.network[0].weight.requires_grad


def test_spatial_runtime_rejects_media_hash_mismatch_before_feature_call(
    tmp_path, monkeypatch
) -> None:
    runtime, _, _ = _runtime(tmp_path, monkeypatch)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(image_path)
    called = False

    def feature_call(*_):
        nonlocal called
        called = True
        return torch.zeros(1, FEATURE_WIDTH)

    monkeypatch.setattr(runtime_module, "videomae_spatial_features", feature_call)
    with pytest.raises(ValueError, match="differs from expected_media_sha256"):
        runtime.decide(
            image_path,
            "color",
            ["red", "blue"],
            expected_media_sha256="0" * 64,
        )
    assert not called


def test_spatial_runtime_rejects_unknown_supplied_option(tmp_path, monkeypatch) -> None:
    runtime, _, _ = _runtime(tmp_path, monkeypatch)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(image_path)

    with pytest.raises(ValueError, match="not valid for taxonomy color"):
        runtime.decide(
            image_path,
            "color",
            ["white", "red"],
            expected_media_sha256=runtime_module.sha256_file(image_path),
        )
