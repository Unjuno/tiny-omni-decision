import numpy as np
import pytest

from tiny_omni_decision import fine_spatial_inputs as module


def test_historical_code_pin_is_not_replaced_by_current_file_hash(tmp_path):
    import hashlib

    p = tmp_path / "producer.py"
    p.write_bytes(b"original")
    pin = hashlib.sha256(p.read_bytes()).hexdigest()
    module.require_pinned_file(p, pin)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.require_pinned_file(p, pin)


def test_fresh_main_reference_uses_actual_config_contract(tmp_path):
    import hashlib
    import json

    ids = [f"video_{i}" for i in range(99)]
    contract = dict(seed=17, plans=[dict(video_id=i) for i in ids])
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cfg = dict(contract=contract, contract_sha256=digest, smoke=False, selected_train_ids=ids)
    config = tmp_path / "config.json"
    config.write_bytes(json.dumps(cfg).encode())
    report = dict(
        status="PASS_ALL_TRAIN_FINE_SPATIAL_CORPUS_NOT_QUALITY",
        contract_sha256=digest,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        features=[dict(video_id=i) for i in ids],
    )
    result = tmp_path / "result.json"
    result.write_bytes(json.dumps(report).encode())
    assert module.require_reference(tmp_path, digest)["status"] == report["status"]
    cfg["contract"]["seed"] = 18
    config.write_bytes(json.dumps(cfg).encode())
    report["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    result.write_bytes(json.dumps(report).encode())
    with pytest.raises(ValueError):
        module.require_reference(tmp_path, digest)
    cfg["contract"]["seed"] = 17
    cfg["selected_train_ids"] = list(reversed(ids))
    report["features"] = [dict(video_id=i) for i in reversed(ids)]
    config.write_bytes(json.dumps(cfg).encode())
    report["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    result.write_bytes(json.dumps(report).encode())
    with pytest.raises(ValueError):
        module.require_reference(tmp_path, digest)


def test_real_fine_cells_keep_low_grid_and_pool_four_actual_high_cells():
    low = np.arange(2 * 32 * 32 * 256, dtype=np.float32).reshape(2, 32, 32, 256)
    high = np.arange(2 * 64 * 64 * 128, dtype=np.float32).reshape(2, 64, 64, 128)
    result = module.fine_context(low, high)
    assert result.shape == (2, 1024, 384) and result.dtype == np.float32
    np.testing.assert_array_equal(result[:, :, :256], low.reshape(2, 1024, 256))
    for t, y, x in [(0, 0, 0), (0, 7, 9), (1, 31, 31)]:
        expected = (
            high[t, y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]
            .mean((0, 1), dtype=np.float64)
            .astype(np.float32)
        )
        np.testing.assert_array_equal(result[t, y * 32 + x, 256:], expected)
    changed = high.copy()
    changed[1, 63, 63, 0] += 32
    delta = module.fine_context(low, changed) - result
    assert np.count_nonzero(delta) == 1 and delta[1, 1023, 256] == 8


@pytest.mark.parametrize("kind", ["coarse", "channels", "time", "integer", "nan"])
def test_native_fine_shapes_and_finite_float32_required(kind):
    low = np.zeros((1, 32, 32, 256), np.float32)
    high = np.zeros((1, 64, 64, 128), np.float32)
    if kind == "coarse":
        low = low[:, ::2, ::2]
    elif kind == "channels":
        high = high[:, :, :, :64]
    elif kind == "time":
        high = np.concatenate([high, high])
    elif kind == "integer":
        low = low.astype(np.int64)
    else:
        high[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        module.fine_context(low, high)


def test_storage_budget_covers_exact_npy_arrays_and_preserves_reserve():
    assert module.fine_array_bytes(1) == 32 * 1024 * (384 + 3) * 4 + 32 * 8 + 3 * 128
    required = module.fine_array_bytes(99) + 8 * 2**30 + 64 * 2**20
    module.require_disk(required, 99)
    with pytest.raises(OSError):
        module.require_disk(required - 1, 99)
    with pytest.raises(ValueError):
        module.fine_array_bytes(True)


def test_smoke_result_and_saved_config_bind_main_contract(tmp_path):
    import hashlib
    import json

    p = tmp_path / "config.json"
    contract = dict(seed=17, state_sha256="frozen")
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    config = dict(
        contract=contract,
        contract_sha256=contract_sha,
        smoke=True,
        selected_train_ids=["video_10103"],
    )
    p.write_bytes(json.dumps(config).encode())
    config_bytes = p.read_bytes()
    pin = hashlib.sha256(p.read_bytes()).hexdigest()
    outputs = {}
    for name in ("appearance", "positions", "frames"):
        dest = tmp_path / (name + ".npy")
        np.save(dest, np.ones(2, np.float32))
        outputs[name] = dict(path=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    report = dict(
        status="PASS_TRAIN_FINE_SPATIAL_SMOKE_NOT_QUALITY",
        contract_sha256=contract_sha,
        config_sha256=pin,
        parameters=31072327,
        first_repeat_bitwise=True,
        state_before="frozen",
        state_after="frozen",
        features=[dict(video_id="video_10103", coarse_reference_bitwise=True, outputs=outputs)],
    )
    (tmp_path / "result.json").write_text(json.dumps(report), encoding="utf-8")
    assert module.require_smoke(tmp_path, contract_sha)["parameters"] == 31072327
    p.write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.require_smoke(tmp_path, contract_sha)
    p.write_bytes(config_bytes)
    with pytest.raises(ValueError):
        module.require_smoke(tmp_path, "other-contract")
    config["contract"]["seed"] = 18
    p.write_bytes(json.dumps(config).encode())
    report["config_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "result.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        module.require_smoke(tmp_path, contract_sha)
    p.write_bytes(config_bytes)
    report["config_sha256"] = pin
    (tmp_path / "result.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "frames.npy").write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.require_smoke(tmp_path, contract_sha)
