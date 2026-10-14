import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.diagnose_condition_binding import verify_original_train  # noqa: E402
from scripts.extract_spatial_context_corpus import sha  # noqa: E402


def fixture(tmp_path):
    path = tmp_path / "epoch.npz"
    values = tuple(np.zeros(shape, np.float32) for shape in ((2, 3), (1, 4), (2, 3)))
    np.savez(path, train=values[0], train_cover=values[1], train_logp=values[2])
    return path, values


def test_original_epoch_hash_and_exact_byte_replay(tmp_path):
    path, values = fixture(tmp_path)
    verify_original_train(path, sha(path), values)
    with pytest.raises(ValueError, match="hash"):
        verify_original_train(path, "changed", values)


@pytest.mark.parametrize("kind", ["dtype", "signed_zero"])
def test_numerically_equal_but_not_bitwise_replay_is_rejected(tmp_path, kind):
    path, values = fixture(tmp_path)
    altered = values[0].astype(np.float64) if kind == "dtype" else values[0].copy()
    if kind == "signed_zero":
        altered[0, 0] = -0.0
    with pytest.raises(ValueError, match="bitwise"):
        verify_original_train(path, sha(path), (altered, *values[1:]))
