import hashlib
import json
import struct

import numpy as np
import pytest


def test_free_atom_has_exact_prefix_size_type_payload_and_no_overwrite(tmp_path):
    from scripts.robomnist_container_features import append_free_atom

    original = tmp_path / "original.mp4"
    derived = tmp_path / "derived.mp4"
    data = b"test-source-prefix"
    original.write_bytes(data)
    r = append_free_atom(original, derived, 16)
    assert original.read_bytes() == data
    assert derived.read_bytes() == data + struct.pack(">I4s", 24, b"free") + bytes(16)
    assert r["source_sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileExistsError):
        append_free_atom(original, derived, 16)
    with pytest.raises(ValueError):
        append_free_atom(original, original, 16)
    with pytest.raises(ValueError):
        append_free_atom(original, tmp_path / "other", 2**32)


def test_both_whole_lenses_letterboxed_without_center_seam_or_view_loss():
    pytest.importorskip("PIL")
    from scripts.robomnist_container_features import stereo_tiles

    a = np.zeros((4, 16, 3), np.uint8)
    a[:, :8] = [255, 0, 0]
    a[:, 8:] = [0, 0, 255]
    tiles = stereo_tiles(a, [0.5, 0.5, 0.5])
    assert tiles.shape == (2, 224, 224, 3)
    assert np.all(tiles[:, :56] == 128)
    assert np.all(tiles[0, 56:168] == [255, 0, 0])
    assert np.all(tiles[1, 56:168] == [0, 0, 255])
    with pytest.raises(ValueError):
        stereo_tiles(np.zeros((4, 15, 3), np.uint8), [0.5] * 3)
    with pytest.raises(ValueError):
        stereo_tiles(a.astype(np.float32), [0.5] * 3)


def test_full_array_identity_includes_dtype_shape_and_every_byte():
    from scripts.robomnist_container_features import digest_arrays, same_arrays

    a = {"x": np.arange(4, dtype=np.float32)}
    same_arrays(a, a)
    assert digest_arrays(a) != digest_arrays({"x": a["x"].astype(np.float64)})
    with pytest.raises(ValueError):
        same_arrays(a, {"x": a["x"] + 1})
    with pytest.raises(ValueError):
        same_arrays(a, {"x": a["x"].reshape(2, 2)})
    with pytest.raises(ValueError):
        same_arrays(a, {})


def test_unavailable_monitor_is_unknown_in_failure_record_not_a_fake_success(monkeypatch):
    from scripts import probe_robomnist_container_features as module

    def unavailable():
        raise RuntimeError("no resource monitor")

    monkeypatch.setattr(module, "peak_bytes", unavailable)
    assert module.failure_peak() is None


def test_terminal_scope_never_admits_model_training_or_cuda():
    from scripts.probe_robomnist_container_features import require_frozen_scope

    r = dict(
        parameters=83101185,
        trainable_parameters=0,
        gradients_present=False,
        model_training=False,
        cuda_initialized=False,
        state_unchanged=True,
        forward_cpu_rng_unchanged=True,
        features_container_invariant=True,
        pixels_container_invariant=True,
        uniform_sensitive=True,
    )
    require_frozen_scope(r)
    with pytest.raises(ValueError):
        require_frozen_scope({**r, "cuda_initialized": True})
    with pytest.raises(ValueError):
        require_frozen_scope({**r, "trainable_parameters": 1})


def test_existing_source_pin_cannot_be_replaced_by_a_later_library_snapshot(tmp_path):
    from scripts.probe_robomnist_container_features import add_pin

    p = tmp_path / "library"
    pins = {str(p.resolve()): "a" * 64}
    add_pin(pins, p, "a" * 64)
    with pytest.raises(ValueError):
        add_pin(pins, p, "b" * 64)


def test_historical_release_is_bound_before_parse_and_weights_are_independently_fixed(monkeypatch):
    from scripts import probe_robomnist_container_features as module

    body = module.R1.read_bytes()
    expected = module.released_r1(body)
    assert expected["model.safetensors"] == module.WEIGHT_SHA
    with pytest.raises(ValueError, match="historical R1"):
        module.released_r1(body + b" ")
    changed = json.loads(body)
    changed["audit"]["input_sha256"]["model.safetensors"] = "a" * 64
    tampered = json.dumps(changed).encode()
    monkeypatch.setattr(module, "R1_SHA", hashlib.sha256(tampered).hexdigest())
    with pytest.raises(ValueError, match="released weight"):
        module.released_r1(tampered)
