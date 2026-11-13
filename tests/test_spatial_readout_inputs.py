import hashlib
import json

import numpy as np
import pytest

from scripts.spatial_readout_inputs import context_tokens, load_spatial_bank
from tiny_omni_decision.spatial_context_features import spatial_context_positions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    plans, rows = [], []
    for i, split in enumerate(("train", "valid")):
        times = np.rint(np.linspace(0, 62, 32)).astype(np.int64)
        plan = dict(
            video_id=f"video_{i}",
            split=split,
            media_sha256=f"media{i}",
            frames=63,
            frame_indices=times.tolist(),
        )
        plans.append(plan)
        arrays = dict(
            appearance=np.full((32, 256, 384), i, np.float32),
            positions=spatial_context_positions(times, source_frames=63),
            frames=times,
        )
        outputs = {}
        for name, array in arrays.items():
            path = tmp_path / f"{split}-video_{i}-{name}.npy"
            np.save(path, array)
            outputs[name] = dict(
                path=str(path),
                sha256=digest(path),
                shape=list(array.shape),
                dtype=str(array.dtype),
                file_bytes=path.stat().st_size,
            )
        rows.append(
            dict(
                video_id=plan["video_id"],
                split=split,
                media_sha256=f"media{i}",
                original_frames=63,
                frame_indices=times.tolist(),
                outputs=outputs,
                selected_rgb_sha256=f"rgb{i}",
                rng_unchanged=True,
            )
        )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(plans=plans, smoke=False, frames=32, grid_size=16)))
    result = dict(
        status="PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY",
        smoke=False,
        features=rows,
        config_sha256=digest(config),
        state_before="frozen",
        state_after="frozen",
        first_repeat_bitwise=True,
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        parameters=31072327,
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    return path, plans, result


def test_complete_hash_bound_corpus_and_mmap(tmp_path):
    path, plans, _ = fixture(tmp_path)
    bank, hashes = load_spatial_bank(path, plans, digest(path))
    assert set(bank) == {"video_0", "video_1"}
    assert len(hashes) == 8
    value = np.load(bank["video_0"]["outputs"]["appearance"]["path"], mmap_mode="r")
    assert isinstance(value, np.memmap) and not value.flags.writeable


@pytest.mark.parametrize("kind", ["partial", "missing", "time", "media", "overlap", "weights"])
def test_partial_or_corrupt_corpus_is_never_training_authority(tmp_path, kind):
    path, plans, result = fixture(tmp_path)
    if kind == "partial":
        result["status"] = "RUNNING"
    elif kind == "missing":
        result["features"].pop()
    elif kind == "time":
        result["features"][0]["frame_indices"][1] = 1
    elif kind == "media":
        result["features"][0]["media_sha256"] = "different"
    elif kind == "overlap":
        result["features"][1]["selected_rgb_sha256"] = "rgb0"
    else:
        result["state_after"] = "changed"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load_spatial_bank(path, plans, digest(path))


def test_file_corruption_cannot_bypass_metadata(tmp_path):
    path, plans, result = fixture(tmp_path)
    output = result["features"][0]["outputs"]["frames"]["path"]
    np.save(output, np.zeros(32, np.int64))
    with pytest.raises(ValueError):
        load_spatial_bank(path, plans, digest(path))


def test_appearance_controls_preserve_positions_and_source(tmp_path):
    path, plans, _ = fixture(tmp_path)
    bank, _ = load_spatial_bank(path, plans, digest(path))
    row = bank["video_0"]
    appearance_path = row["outputs"]["appearance"]["path"]
    original = np.arange(32, dtype=np.float32)[:, None, None] * np.ones((32, 256, 384), np.float32)
    np.save(appearance_path, original)
    temporal = context_tokens(row, "temporal_appearance").reshape(32, 256, 387)
    first = context_tokens(row, "initial_broadcast").reshape(32, 256, 387)
    reverse = context_tokens(row, "temporal_appearance", "reverse").reshape(32, 256, 387)
    np.testing.assert_array_equal(temporal[..., 384:], first[..., 384:])
    np.testing.assert_array_equal(temporal[..., 384:], reverse[..., 384:])
    np.testing.assert_array_equal(first[..., :384], np.broadcast_to(original[:1], original.shape))
    np.testing.assert_array_equal(reverse[..., :384], original[::-1])
    np.testing.assert_array_equal(np.load(appearance_path), original)
    with pytest.raises(ValueError):
        context_tokens(row, "unknown")
