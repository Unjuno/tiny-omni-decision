import numpy as np
import pytest

from tiny_omni_decision import mask_memory_probe as probe


def test_seed_cohort_retains_every_nms_mask_in_original_order():
    masks = np.zeros((4, 256, 256), bool)
    for i in range(4):
        masks[i, i : i + 2, i : i + 2] = True
    packed = np.packbits(masks.reshape(4, -1), axis=1, bitorder="big")
    indices = np.array([3, 0, 2], np.int64)
    np.testing.assert_array_equal(probe.seed_masks(packed, indices), masks[indices])
    with pytest.raises(ValueError):
        probe.seed_masks(packed, np.array([0, 0], np.int64))
    with pytest.raises(ValueError):
        probe.seed_masks(packed[:, :-1], indices)


def test_static_input_changes_rgb_only_and_original_frames_are_not_mutated():
    rgb = np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3)
    before = rgb.tobytes()
    np.testing.assert_array_equal(probe.video_control(rgb, "actual"), rgb)
    repeated = probe.video_control(rgb, "initial_repeat")
    np.testing.assert_array_equal(repeated, np.broadcast_to(rgb[:1], rgb.shape))
    assert rgb.tobytes() == before
    with pytest.raises(ValueError):
        probe.video_control(rgb, "other")


def test_stream_outputs_keep_empty_masks_and_fixed_shape_dtype():
    logits = np.full((1, 1, 256, 256), -2, np.float32)
    logits[0, 0, 2:4, 5:8] = 2
    p, b = probe.pack_frame(logits, np.array([[-1]], np.float32))
    assert p.shape == (8192,) and p.dtype == np.uint8
    np.testing.assert_array_equal(b, np.array([5 / 256, 2 / 256, 8 / 256, 4 / 256], np.float64))
    empty, b = probe.pack_frame(np.full_like(logits, -2), np.array([[-1]], np.float32))
    assert not empty.any() and not b.any()
    with pytest.raises(ValueError):
        probe.pack_frame(logits.astype(float), np.array([[-1]], np.float32))
    with pytest.raises(ValueError):
        probe.pack_frame(logits, np.array([-1], np.float32))


def test_source_time_contract_and_bad_cut_fail_closed():
    frames = np.rint(np.linspace(0, 458, 32)).astype(np.int64)
    probe.verify_times(frames, 459, 459)
    probe.verify_times(frames, 459, -1)
    with pytest.raises(ValueError):
        probe.verify_times(frames, 459, 458)
    with pytest.raises(ValueError):
        probe.verify_times(frames[::-1], 459, 459)
