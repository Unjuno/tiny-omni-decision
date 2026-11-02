from __future__ import annotations

import numpy as np
import pytest

from scripts.complete_avatar_category_transfer_artifacts import (
    fit_normalization,
    write_supplement,
)


def test_fit_normalization_uses_float32_fit_features_and_std_floor():
    audio = np.zeros((3, 384), dtype=np.float32)
    audio[:, 0] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    video = np.zeros((3, 4, 768), dtype=np.float32)
    video[:, :, 0] = np.asarray([[1.0] * 4, [2.0] * 4, [3.0] * 4], dtype=np.float32)

    result = fit_normalization(audio, video)

    assert all(value.dtype == np.float32 for value in result.values())
    assert result["audio_mean"][0] == pytest.approx(2.0)
    assert result["video_mean"][0] == pytest.approx(2.0)
    assert result["audio_std"][1] == pytest.approx(1e-5)
    assert result["video_std"][1] == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("audio", "video", "message"),
    [
        (np.zeros((2, 383), dtype=np.float32), np.zeros((2, 4, 768), dtype=np.float32), "audio"),
        (np.zeros((2, 384), dtype=np.float32), np.zeros((2, 3, 768), dtype=np.float32), "video"),
        (np.zeros((0, 384), dtype=np.float32), np.zeros((0, 4, 768), dtype=np.float32), "empty"),
    ],
)
def test_fit_normalization_rejects_invalid_shapes(audio, video, message):
    with pytest.raises(ValueError, match=message):
        fit_normalization(audio, video)


def test_supplement_writer_refuses_to_overwrite(tmp_path):
    (tmp_path / "normalization.npz").write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_supplement(tmp_path, b"new", {"example": True})

    assert (tmp_path / "normalization.npz").read_bytes() == b"preserve"
    assert not (tmp_path / "artifact-completion.json").exists()
