import hashlib

import pytest

from scripts import extract_mask_feature_corpus as runner


def test_protected_pilot_config_cannot_be_silently_repinned(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"seed":17}\n')
    report = {"config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    runner.require_config_binding(path, report)
    path.write_bytes(b'{"seed":18}\n')
    with pytest.raises(ValueError):
        runner.require_config_binding(path, report)


def test_final_body_at_limit_is_allowed_but_one_over_limit_fails():
    runner.require_resource_limits(600.0, 8 * 2**30)
    with pytest.raises(RuntimeError):
        runner.require_resource_limits(600.0001, 8 * 2**30)


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_invalid_elapsed_measurement_fails_closed(elapsed):
    with pytest.raises(RuntimeError):
        runner.require_resource_limits(elapsed, 8 * 2**30)


def test_final_disk_reserve_cannot_be_consumed():
    with pytest.raises(RuntimeError):
        runner.require_resource_limits(1, 8 * 2**30 - 1)
