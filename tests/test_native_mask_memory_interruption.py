import numpy as np
import pytest


def partial():
    return dict(
        masks_packed=np.zeros((1, 32, 8192), np.uint8),
        boxes=np.zeros((1, 32, 4), np.float64),
        scores=np.zeros((1, 32), np.float32),
        empty_masks=np.ones((1, 32), bool),
        raw_sha256=np.array(["a" * 64], dtype="<U64"),
    )


def test_partial_checks_geometry_but_does_not_invent_stored_identity():
    from scripts.record_native_mask_memory_interruption import verify_partial

    r = verify_partial(partial(), 1)
    assert r["native_object_frames"] == 32
    assert r["independently_stored_seed_frame_identity"] is False
    assert r["raw_logits_reconstructed"] is False


def test_partial_count_and_geometry_corruption_rejected():
    from scripts.record_native_mask_memory_interruption import verify_partial

    with pytest.raises(ValueError):
        verify_partial(partial(), 2)
    a = partial()
    a["boxes"][0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        verify_partial(a, 1)


def test_partial_extra_identity_cannot_be_silently_discarded():
    from scripts.record_native_mask_memory_interruption import verify_partial

    a = partial()
    a["seed_indices"] = np.array([4])
    with pytest.raises(ValueError):
        verify_partial(a, 1)


def test_incomplete_rows_must_be_the_exact_ordered_cohort_prefix():
    from scripts.record_native_mask_memory_interruption import require_prefix

    plans = [{"video_id": "v1"}, {"video_id": "v2"}]
    rows = [{"video_id": "v1", "arm": "actual"}, {"video_id": "v1", "arm": "initial_repeat"}]
    assert require_prefix(rows, plans) == 2
    with pytest.raises(ValueError):
        require_prefix(list(reversed(rows)), plans)
    with pytest.raises(ValueError):
        require_prefix(rows * 2, plans)
