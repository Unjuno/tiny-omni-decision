import numpy as np
import pytest

from scripts import fine_qa_bank as module


def test_train_filter_never_resolves_validation_arrays():
    rows = [
        dict(video_id="train1", split="train", path="train.npy"),
        dict(video_id="valid1", split="valid", path="do-not-open.npy"),
    ]
    assert module.train_rows(rows, ["train1"]) == {"train1": rows[0]}
    with pytest.raises(ValueError):
        module.train_rows(rows + [rows[0]], ["train1"])
    with pytest.raises(ValueError):
        module.train_rows(rows, ["missing"])


def test_native_time_and_media_binding_rejects_wrong_clip_or_frame():
    row = dict(
        video_id="x",
        split="train",
        media_sha256="abc",
        original_frames=33,
        frame_indices=list(range(32)),
        selected_rgb_sha256="rgb",
    )
    point = dict(row, frames=33)
    media = dict(video_id="x", sha256="abc", cut_frame=33)
    module.verify_binding(row, dict(row), point, media)
    module.verify_binding(row, dict(row), point, dict(media, cut_frame=-1))
    with pytest.raises(ValueError):
        module.verify_binding(row, dict(row), point, dict(media, cut_frame=34))
    for key, value in [
        ("media_sha256", "other"),
        ("frame_indices", list(range(1, 33))),
        ("selected_rgb_sha256", "other"),
    ]:
        bad = dict(row, **{key: value})
        with pytest.raises(ValueError):
            module.verify_binding(row, bad, point, media)


def test_array_schema_is_strict_and_memory_mapped(tmp_path):
    path = tmp_path / "native.npy"
    np.save(path, np.zeros((2, 4, 384), np.float32))
    row = dict(
        path=str(path),
        sha256=module.sha(path),
        shape=[2, 4, 384],
        dtype="float32",
        file_bytes=path.stat().st_size,
    )
    hashes = {}
    value = module.native_array(row, (2, 4, 384), np.float32, hashes)
    assert isinstance(value, np.memmap) and hashes[str(path)] == row["sha256"]
    with pytest.raises(ValueError):
        module.native_array(row, (2, 5, 384), np.float32, {})
    del value  # Windows cannot replace an open memory-mapped file.
    np.save(path, np.full((2, 4, 384), np.nan, np.float32))
    row["sha256"] = module.sha(path)
    with pytest.raises(ValueError):
        module.native_array(row, (2, 4, 384), np.float32, {})
