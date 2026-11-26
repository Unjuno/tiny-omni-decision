import numpy as np
import pytest

from scripts.audit_native_mask_memory_full import require_reference_bindings, verify_packed_archive


def archive():
    mask = np.zeros((2, 256, 256), bool)
    mask[0, 2:4, 1:3] = True
    return dict(
        masks_packed=np.packbits(mask.reshape(1, 2, -1), axis=-1, bitorder="big"),
        boxes=np.array([[[1 / 256, 2 / 256, 3 / 256, 4 / 256], [0, 0, 0, 0]]], np.float64),
        scores=np.zeros((1, 2), np.float32),
        empty_masks=np.array([[False, True]]),
        raw_sha256=np.array(["a" * 64]),
        seed_indices=np.array([7], np.int64),
        frame_indices=np.array([0, 4], np.int64),
    )


def test_complete_packed_geometry_and_empty_accounting():
    assert verify_packed_archive(archive(), [7], [0, 4]) == dict(
        native_object_frames=2, empty_masks=1
    )


def test_wrong_ids_mask_geometry_score_dtype_and_digest_are_rejected():
    for key, value in [
        ("seed_indices", np.array([8], np.int64)),
        ("boxes", np.ones((1, 2, 4), np.float64)),
        ("scores", np.zeros((1, 2), np.float64)),
        ("empty_masks", np.array([[True, True]])),
        ("raw_sha256", np.array(["z" * 64])),
    ]:
        a = archive()
        a[key] = value
        with pytest.raises(ValueError):
            verify_packed_archive(a, [7], [0, 4])


def test_fresh_references_must_bind_main_config_result_and_all_archives():
    expected = {
        "main/result.json": "a" * 64,
        "main/config.json": "b" * 64,
        "main/all.npz": "c" * 64,
    }
    assert require_reference_bindings(expected, dict(expected)) == expected
    for key in expected:
        missing = dict(expected)
        missing.pop(key)
        with pytest.raises(ValueError):
            require_reference_bindings(expected, missing)
    changed = dict(expected, **{"main/all.npz": "d" * 64})
    with pytest.raises(ValueError):
        require_reference_bindings(expected, changed)
