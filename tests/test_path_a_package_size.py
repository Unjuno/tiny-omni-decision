from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_path_a_package_size import (
    _canonical_numpy_dtype,
    _estimate_int8,
    _listed_file,
)


def test_int8_estimate_includes_group_metadata_preserved_tensors_and_file_overhead() -> None:
    estimate = _estimate_int8(
        [
            {"dtype": "F32", "elements": 257, "element_bytes": 4},
            {"dtype": "I64", "elements": 1, "element_bytes": 8},
        ],
        group_size=256,
        fixed_package_bytes=100,
        encoder_file_bytes=1_039,
    )

    assert estimate == {
        "group_size": 256,
        "float32_encoder_elements": 257,
        "non_float32_encoder_elements": 1,
        "group_count": 2,
        "int8_payload_bytes": 257,
        "float32_scale_bytes": 8,
        "int8_zero_point_bytes": 2,
        "other_encoder_payload_bytes_preserved": 8,
        "existing_encoder_file_overhead_preserved": 3,
        "fixed_package_bytes": 100,
        "estimated_package_bytes": 378,
    }


def test_int8_estimate_rejects_group_size_zero() -> None:
    with pytest.raises(ValueError, match="group size must be positive"):
        _estimate_int8([], group_size=0, fixed_package_bytes=0, encoder_file_bytes=0)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(np.dtype("float32"), "F32"), (np.dtype("int64"), "I64"), (np.dtype("uint8"), "U8")],
)
def test_numpy_dtypes_are_normalized_to_safetensors_names(dtype, expected: str) -> None:
    assert _canonical_numpy_dtype(dtype) == expected


def test_listed_package_path_rejects_parent_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsafe package path"):
        _listed_file(tmp_path, "../outside.bin")
