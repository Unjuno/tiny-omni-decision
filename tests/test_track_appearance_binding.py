import numpy as np
import pytest

from scripts.probe_track_appearance import validate_binding


def fixture():
    times = np.rint(np.linspace(0, 62, 32)).astype(np.int64)
    point = dict(
        video_id="video_1",
        split="train",
        frames=63,
        shape=[256, 512],
        media_sha256="media",
        frame_indices=times.tolist(),
    )
    plan = dict(video_id="video_1", split="train", frames=63, media_sha256="media")
    row = dict(video_id="video_1", original_clip_frames=63, frames_encoded=32)
    return point, plan, row, times


def test_exact_same_media_frames_and_training_identity_required():
    point, plan, row, times = fixture()
    validate_binding(point, plan, row, times)


@pytest.mark.parametrize("kind", ["split", "media", "time", "identity", "shape", "frames"])
def test_unaligned_cache_cannot_be_sampled(kind):
    point, plan, row, times = fixture()
    if kind == "split":
        point["split"] = "valid"
    elif kind == "media":
        point["media_sha256"] = "other"
    elif kind == "time":
        times[1] = 1
    elif kind == "identity":
        row["video_id"] = "video_2"
    elif kind == "shape":
        point["shape"] = [255, 512]
    else:
        plan["frames"] = 62
    with pytest.raises(ValueError):
        validate_binding(point, plan, row, times)
