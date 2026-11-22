from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.build_videoprism_fp16_storage_bundle import (
    convert_npz_to_float16,
)


def test_convert_npz_to_float16_preserves_names_shapes_and_rounded_values(tmp_path):
    source = tmp_path / "source.npz"
    destination = tmp_path / "converted.npz"
    weights = {
        "params/a": np.asarray([[0.5, -1.25]], dtype=np.float32),
        "params/b": np.asarray([2.0], dtype=np.float32),
    }
    np.savez(source, **weights)

    parameters, tensors = convert_npz_to_float16(source, destination)

    assert (parameters, tensors) == (3, 2)
    with np.load(destination, allow_pickle=False) as archive:
        assert set(archive.files) == set(weights)
        for key, original in weights.items():
            assert archive[key].shape == original.shape
            assert archive[key].dtype == np.float16
            np.testing.assert_array_equal(archive[key], original.astype(np.float16))


@pytest.mark.parametrize("value", [np.asarray([np.nan], dtype=np.float32),
                                    np.asarray([1.0], dtype=np.float64)])
def test_convert_npz_to_float16_rejects_invalid_source_values(tmp_path, value):
    source = tmp_path / "source.npz"
    destination = tmp_path / "converted.npz"
    np.savez(source, tensor=value)

    with pytest.raises(ValueError, match="finite float32"):
        convert_npz_to_float16(source, destination)


def test_convert_npz_to_float16_rejects_half_precision_overflow(tmp_path):
    source = tmp_path / "source.npz"
    destination = tmp_path / "converted.npz"
    np.savez(source, tensor=np.asarray([1e8], dtype=np.float32))

    with pytest.raises(ValueError, match="overflowed"):
        convert_npz_to_float16(source, destination)
