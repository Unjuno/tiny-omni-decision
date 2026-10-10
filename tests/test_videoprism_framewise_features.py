from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from scripts.extract_videoprism_framewise_features import (  # noqa: E402
    _unique_asset_rows,
    _validate_cross_sample,
    center_framewise_features,
    pool_framewise_tokens,
)


def test_framewise_pooling_preserves_time_and_matches_global_mean() -> None:
    tokens = np.arange(1 * 8 * 3 * 768, dtype=np.float32).reshape(1, 8 * 3, 768)

    framewise = pool_framewise_tokens(tokens)

    assert framewise.shape == (8, 768)
    np.testing.assert_allclose(framewise.mean(axis=0), tokens.mean(axis=1)[0], atol=1e-4)
    assert not np.array_equal(framewise[0], framewise[-1])


@pytest.mark.parametrize(
    "tokens",
    [
        np.zeros((8 * 3, 768), dtype=np.float32),
        np.zeros((1, 8 * 3, 767), dtype=np.float32),
        np.zeros((1, 8 * 3 + 1, 768), dtype=np.float32),
    ],
)
def test_framewise_pooling_rejects_incompatible_token_shapes(tokens: object) -> None:
    with pytest.raises(ValueError):
        pool_framewise_tokens(tokens)  # type: ignore[arg-type]


def test_centering_reproduces_reference_mean_without_changing_time_differences() -> None:
    framewise = np.random.default_rng(17).normal(0, 0.5, size=(8, 768)).astype(np.float32)
    reference = framewise.mean(axis=0) + np.linspace(-3e-5, 3e-5, 768, dtype=np.float32)
    before_differences = framewise[1:] - framewise[:-1]

    centered, delta = center_framewise_features(framewise, reference)

    assert delta <= 1e-5
    np.testing.assert_allclose(centered.mean(axis=0), reference, atol=1e-5)
    np.testing.assert_allclose(centered[1:] - centered[:-1], before_differences, atol=1e-5)


def test_cross_sample_validation_rejects_scene_and_media_overlap() -> None:
    train = [{"id": "train", "split": "train", "scene_index": 1, "media_sha256": "a"}]
    evaluation = [
        {"id": "eval", "split": "validation", "scene_index": 1, "media_sha256": "b"}
    ]

    with pytest.raises(ValueError, match="scene or media identity overlap"):
        _validate_cross_sample(train, evaluation)


def test_unique_asset_rows_deduplicates_multiple_questions(tmp_path) -> None:
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixed-test-video")
    from tiny_omni_decision.dataset import sha256_file

    digest = sha256_file(media)
    rows = [
        {"id": f"q{index}", "media_path": media.name, "media_sha256": digest}
        for index in range(2)
    ]

    assets = _unique_asset_rows([(tmp_path, rows)])

    assert list(assets) == [digest]


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.evaluate_videoprism_temporal_pooling_control import _weighted_pool  # noqa: E402


def test_uniform_temporal_weights_reproduce_all_frame_average() -> None:
    frame_features = torch.arange(2 * 8 * 768, dtype=torch.float32).reshape(2, 8, 768)
    weights = torch.full((8,), 1 / 8)

    pooled = _weighted_pool(frame_features, weights)

    assert pooled.shape == (2, 768)
    torch.testing.assert_close(pooled, frame_features.mean(dim=1))


def test_weighted_pool_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="frame features"):
        _weighted_pool(torch.zeros(2, 7, 768), torch.full((8,), 1 / 8))
