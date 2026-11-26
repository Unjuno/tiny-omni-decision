import numpy as np
import pytest

from scripts.probe_dynamic_discovery import input_plan, verify_cache_policy


def test_cache_admission_rejects_changed_processor_or_policy():
    c = {
        k: "same"
        for k in (
            "input_plan",
            "source_hashes",
            "cache_policy",
            "model_revision",
            "grid_side",
            "chunk_objects",
            "mask_threshold",
            "nms_iou",
            "top_k",
            "dtype",
            "attention",
            "tf32",
            "fine_time_archive_sha256",
        )
    }
    verify_cache_policy(c, c.copy())
    for key in ("source_hashes", "cache_policy"):
        other = c.copy()
        other[key] = "changed"
        with pytest.raises(ValueError):
            verify_cache_policy(c, other)


def plan():
    return dict(
        video_id="video_10903",
        split="train",
        media_path="clip.mp4",
        media_sha256="hash",
        cut_frame=558,
        frames=558,
        query_path="not-loaded",
        grounded_role=7,
    )


def test_sanitized_plan_retains_same32_uniform_input_times_without_label_fields():
    times = np.rint(np.linspace(0, 557, 32)).astype(np.int64)
    output = input_plan(plan(), times)
    assert set(output) == {
        "video_id",
        "split",
        "media_path",
        "media_sha256",
        "cut_frame",
        "frames",
        "frame_indices",
    }
    assert output["frame_indices"] == times.tolist()
    assert "grounded_role" not in output and "query_path" not in output


@pytest.mark.parametrize("kind", ["split", "cut", "times"])
def test_changed_input_split_cut_or_time_fails_closed(kind):
    p = plan()
    times = np.rint(np.linspace(0, 557, 32)).astype(np.int64)
    if kind == "split":
        p["split"] = "valid"
    elif kind == "cut":
        p["cut_frame"] = 557
    else:
        times[1] += 1
    with pytest.raises(ValueError):
        input_plan(p, times)
