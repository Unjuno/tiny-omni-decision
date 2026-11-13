import io

import numpy as np
import pytest

from tiny_omni_decision.token_transport import read_tokens, write_tokens


def test_roundtrip_exact_and_eof():
    stream = io.BytesIO()
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    write_tokens(stream, "asset", values)
    stream.seek(0)
    key, received = read_tokens(stream, values.shape)
    assert key == "asset"
    np.testing.assert_array_equal(received, values)
    assert read_tokens(stream, values.shape) is None


def test_reject_corruption_truncation_and_wrong_shape():
    stream = io.BytesIO()
    write_tokens(stream, "asset", np.ones((2, 3), dtype=np.float32))
    raw = stream.getvalue()
    for damaged in [raw[:-1], raw[:-1] + bytes([raw[-1] ^ 1])]:
        with pytest.raises(ValueError):
            read_tokens(io.BytesIO(damaged), (2, 3))
    with pytest.raises(ValueError):
        read_tokens(io.BytesIO(raw), (3, 2))


def test_reject_nonfinite_dtype_and_oversized_header():
    for values in [np.array([np.nan], dtype=np.float32), np.ones(2, dtype=np.float16)]:
        with pytest.raises(ValueError):
            write_tokens(io.BytesIO(), "asset", values)
    with pytest.raises(ValueError):
        read_tokens(io.BytesIO(b"x" * 4097), (2,))
