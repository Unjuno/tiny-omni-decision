"""Tests for DextraVerse feature cache transformations."""

from __future__ import annotations

import sys
from io import BytesIO

import numpy as np
import pytest

from scripts.pretrained_reuse import extract_dextraverse_contact_features
from scripts.pretrained_reuse.dextraverse_features import (
    assemble_sensor_window,
    pool_videoprism_tokens,
    read_video_batch,
    read_video_features,
    write_video_batch,
    write_video_features,
)


def test_spatial_token_pooling_preserves_eight_temporal_positions() -> None:
    tokens = np.arange(8 * 4 * 768, dtype=np.float32).reshape(1, 8 * 4, 768)
    pooled = pool_videoprism_tokens(tokens)
    assert pooled.shape == (8, 768)
    np.testing.assert_allclose(pooled[0], tokens.reshape(1, 8, 4, 768)[0, 0].mean(0))
    assert not np.array_equal(pooled[0], pooled[-1])


def test_sensor_window_keeps_left_then_right_channels_and_causal_time() -> None:
    left = np.zeros((5, 5, 3), dtype=np.float32)
    right = np.zeros_like(left)
    left[:, 0, 0] = 1
    right[:, 0, 0] = 2
    times = np.arange(5, dtype=np.float64) * 0.1
    values, relative_ms = assemble_sensor_window(left, right, times, [2, 3, 4], 4)
    assert values.shape == (3, 30)
    assert values[0, 0] == 1
    assert values[0, 15] == 2
    np.testing.assert_array_equal(relative_ms, [-200, -100, 0])


def test_sensor_window_rejects_noncausal_query() -> None:
    force = np.zeros((4, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="causal"):
        assemble_sensor_window(force, force, np.arange(4), [0, 1, 2], 1)


def test_video_pool_rejects_wrong_layout_and_nonfinite_tokens() -> None:
    with pytest.raises(ValueError, match="VideoPrism"):
        pool_videoprism_tokens(np.zeros((1, 7, 768), dtype=np.float32))
    tokens = np.zeros((1, 8, 768), dtype=np.float32)
    tokens[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pool_videoprism_tokens(tokens)


def test_binary_encoder_pipe_round_trips_rgb_and_temporal_features() -> None:
    frames = np.arange(2 * 8 * 240 * 320 * 3, dtype=np.uint8).reshape(
        2, 8, 240, 320, 3
    )
    features = np.zeros((2, 8, 768), dtype=np.float32)
    features[1, :, 0] = np.arange(8, dtype=np.float32)
    source = BytesIO()
    assert write_video_batch(source, frames) == 2
    write_video_features(source, features)
    source.write(b"\x00\x00\x00\x00")
    source.seek(0)
    np.testing.assert_array_equal(read_video_batch(source), frames)
    np.testing.assert_array_equal(read_video_features(source, 2), features)
    assert read_video_batch(source) is None


def test_binary_encoder_pipe_rejects_bad_batch_and_truncated_feature_response() -> None:
    with pytest.raises(ValueError, match="uint8"):
        write_video_batch(BytesIO(), np.zeros((1, 8, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(EOFError, match="closed"):
        read_video_features(BytesIO(b""), 1)


def test_extractor_cli_forwards_required_videoprism_runtime(monkeypatch, tmp_path) -> None:
    output = tmp_path / "new-cache"
    arguments = [
        "extract_dextraverse_contact_features.py",
        "--archive", str(tmp_path / "source.zip"),
        "--preflight", str(tmp_path / "preflight.json"),
        "--checkpoint", str(tmp_path / "videoprism.npz"),
        "--videoprism-source", str(tmp_path / "videoprism-source"),
        "--videoprism-python", str(tmp_path / "videoprism-python.exe"),
        "--output", str(output),
        "--device", "cpu",
        "--batch-size", "3",
    ]
    captured = {}

    def fake_extract(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "smoke"}

    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        extract_dextraverse_contact_features, "extract_features", fake_extract
    )
    extract_dextraverse_contact_features.main()
    assert captured["args"][4] == tmp_path / "videoprism-python.exe"
    assert captured["args"][5] == output
    assert captured["kwargs"] == {"device": "cpu", "batch_size": 3}


def test_archive_group_resolves_root_and_single_wrapper_directory() -> None:
    from scripts.pretrained_reuse.inspect_dextraverse_contact_archive import (
        open_archive_group,
    )

    class Store:
        def __init__(self, keys):
            self._keys = keys

        def keys(self):
            return self._keys

    class Zarr:
        @staticmethod
        def open_group(*, store, path, mode):
            return store, path, mode

    root_store = Store([".zgroup", "data/.zgroup"])
    assert open_archive_group(root_store, Zarr) == (root_store, "", "r")
    wrapped_store = Store([
        "pick_cube_place_box.zarr/.zgroup",
        "pick_cube_place_box.zarr/data/.zgroup",
    ])
    assert open_archive_group(wrapped_store, Zarr) == (
        wrapped_store, "pick_cube_place_box.zarr", "r"
    )


def test_archive_group_rejects_ambiguous_top_level_groups() -> None:
    from scripts.pretrained_reuse.inspect_dextraverse_contact_archive import (
        open_archive_group,
    )

    class Store:
        def keys(self):
            return ["first.zarr/.zgroup", "second.zarr/.zgroup"]

    class Zarr:
        @staticmethod
        def open_group(**kwargs):
            raise AssertionError("ambiguous archive must fail before opening a group")

    with pytest.raises(ValueError, match="one root Zarr group"):
        open_archive_group(Store(), Zarr)
