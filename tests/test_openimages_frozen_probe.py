import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("transformers")

from scripts.train_openimages_frozen_probe import (  # noqa: E402
    _quality,
    _tasks,
    _validate_rows,
)


def _row(image_id, target, target_label):
    return {
        "image_id": image_id,
        "sample_id": f"sample:{image_id}",
        "media_sha256": f"media:{image_id}",
        "normalized_rgb_sha256": f"rgb:{image_id}",
        "original_md5_base64": f"md5:{image_id}",
        "target": target,
        "target_label": target_label,
        "options": [target, "B", "C", "D"],
    }


def test_candidate_features_keep_option_order_and_target_position():
    rows = [_row("image-1", "C", "/class/c")]
    rows[0]["options"] = ["A", "B", "C", "D"]
    images = np.zeros((1, 768), dtype="float32")
    texts = np.zeros((1, 4, 384), dtype="float32")

    features, targets = _tasks(images, texts, rows)

    assert features.shape == (1, 4, 1152)
    assert targets.tolist() == [2]


def test_uniform_candidate_quality_has_expected_calibration_baseline():
    logits = torch.zeros((4, 4))
    targets = torch.tensor([0, 1, 2, 3])

    metrics = _quality(logits, targets)

    assert metrics["accuracy"] == 0.25
    assert metrics["nll"] == pytest.approx(1.38629436)
    assert metrics["brier"] == pytest.approx(0.75)
    assert metrics["mean_confidence"] == 0.25


def test_split_validation_rejects_media_identity_overlap():
    train = [_row("train-image", "A", "/class/a")]
    validation = [_row("val-image", "A", "/class/a")]
    validation[0]["normalized_rgb_sha256"] = train[0]["normalized_rgb_sha256"]

    with pytest.raises(ValueError, match="normalized_rgb_sha256 overlap"):
        _validate_rows(train, validation)
