"""Installed CPU session contract; no model weights or GPU work."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
Sam2VideoInferenceSession = pytest.importorskip(
    "transformers.models.sam2_video.modeling_sam2_video"
).Sam2VideoInferenceSession


def test_tracking_reset_keeps_same_frame_and_vision_cache_and_removes_object_history():
    frame = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    s = Sam2VideoInferenceSession(video=frame, inference_device="cpu", dtype=torch.float32)
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    s.cache.cache_vision_features(0, dict(vision_feats=feature, vision_pos_embeds=[feature + 1]))
    idx = s.obj_id_to_idx(7)
    s.add_point_inputs(idx, 0, dict(point_coords=torch.zeros(1, 2)))
    s.mask_inputs_per_obj[idx][0] = torch.ones(2, 2)
    s.output_dict_per_obj[idx]["cond_frame_outputs"][0] = dict(dummy=feature)
    s.frames_tracked_per_obj[idx][0] = dict(reverse=False)
    s.obj_with_new_inputs = [idx]
    pixels = s.get_frame(0).clone()
    cached = s.cache.get_vision_features(0)
    s.reset_tracking_data()
    assert s.num_frames == 1 and s.get_obj_num() == 0 and s.obj_ids == []
    assert not s.point_inputs_per_obj and not s.mask_inputs_per_obj
    assert not s.output_dict_per_obj and not s.frames_tracked_per_obj and not s.obj_with_new_inputs
    assert torch.equal(pixels, s.get_frame(0))
    assert torch.equal(cached["vision_feats"], s.cache.get_vision_features(0)["vision_feats"])
    assert torch.equal(
        cached["vision_pos_embeds"][0], s.cache.get_vision_features(0)["vision_pos_embeds"][0]
    )
    assert s.obj_id_to_idx(99) == 0


def test_distinct_source_frame_session_never_inherits_vision_cache():
    a = Sam2VideoInferenceSession(
        video=torch.zeros(1, 3, 8, 8), inference_device="cpu", dtype=torch.float32
    )
    b = Sam2VideoInferenceSession(
        video=torch.ones(1, 3, 8, 8), inference_device="cpu", dtype=torch.float32
    )
    a.cache.cache_vision_features(0, dict(vision_feats=torch.zeros(1, 2, 2, 2)))
    a.reset_tracking_data()
    assert a.cache.get_vision_features(0) is not None
    assert b.cache.get_vision_features(0) is None
    assert not torch.equal(a.get_frame(0), b.get_frame(0))
