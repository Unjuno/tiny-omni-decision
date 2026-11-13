from copy import deepcopy

import pytest

from scripts.native_full_extent_binding import bind_visual_banks


def banks():
    original = {
        "image/train-cells.npy": {
            "path": "/old/image.npy",
            "sha256": "a" * 64,
            "shape": [8, 17, 768],
        },
        "video/scene.npy": {"path": "/old/video.npy", "sha256": "b" * 64, "shape": [2048, 768]},
        "audio/tokens.npy": {"path": "/old/audio.npy", "sha256": "c" * 64, "shape": [100, 384]},
    }
    replacement = {
        k: {**v, "path": "/new/" + k.split("/")[1], "sha256": "d" * 64}
        for k, v in original.items()
        if k.startswith(("image/", "video/"))
    }
    return original, replacement


def test_only_visual_bindings_change_without_mutating_inputs():
    original, replacement = banks()
    before = deepcopy(original)
    result = bind_visual_banks(original, replacement, "/new")
    assert original == before
    assert result["audio/tokens.npy"] == original["audio/tokens.npy"]
    assert result["image/train-cells.npy"] == replacement["image/train-cells.npy"]
    assert result["video/scene.npy"] == replacement["video/scene.npy"]
    assert set(result) == set(original)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "shape", "escape", "old_path", "alias", "no_change", "bad_hash"]
)
def test_invalid_overlay_is_rejected(change):
    original, replacement = banks()
    image = replacement["image/train-cells.npy"]
    if change == "missing":
        del replacement["video/scene.npy"]
    elif change == "extra":
        replacement["audio/tokens.npy"] = original["audio/tokens.npy"]
    elif change == "shape":
        image["shape"] = [8, 16, 768]
    elif change == "escape":
        image["path"] = "/new/../old/image.npy"
    elif change == "old_path":
        image["path"] = original["image/train-cells.npy"]["path"]
    elif change == "alias":
        image["path"] = replacement["video/scene.npy"]["path"]
    elif change == "no_change":
        for k in replacement:
            replacement[k]["sha256"] = original[k]["sha256"]
    elif change == "bad_hash":
        image["sha256"] = "UNKNOWN"
    with pytest.raises(ValueError):
        bind_visual_banks(original, replacement, "/new")
