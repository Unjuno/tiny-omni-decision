import json
from pathlib import Path

import pytest

from scripts.hotgpu_preflight import validate_gpu_preflight


def test_unreserved_launch_requires_an_idle_gpu_without_compute_pids():
    result = validate_gpu_preflight(
        memory_used_mib=100,
        utilization_pct=0,
        total_mib=16384,
        active_pids=set(),
        own_pid=10,
        environment={},
    )

    assert result["mode"] == "unreserved_idle"
    with pytest.raises(RuntimeError, match="GPU occupied"):
        validate_gpu_preflight(
            memory_used_mib=100,
            utilization_pct=0,
            total_mib=16384,
            active_pids={20},
            own_pid=10,
            environment={},
        )


def test_hotgpu_launch_accepts_only_its_reserved_parent_and_child():
    environment = {
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
        "TINY_OMNI_GPU_GATE_PARENT_PID": "10",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": "14000",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": "8000",
    }
    result = validate_gpu_preflight(
        memory_used_mib=3200,
        utilization_pct=0,
        total_mib=16384,
        active_pids={10, 11},
        own_pid=11,
        environment=environment,
    )

    assert result == {
        "mode": "hotgpu_reserved",
        "parent_pid": 10,
        "reserved_mib": 3200,
        "memory_used_mib": 3200,
        "free_mib": 13184,
        "utilization_pct": 0,
        "active_compute_pids": 2,
    }
    with pytest.raises(RuntimeError, match="handoff"):
        validate_gpu_preflight(
            memory_used_mib=3200,
            utilization_pct=0,
            total_mib=16384,
            active_pids={10, 11, 12},
            own_pid=11,
            environment=environment,
        )


@pytest.mark.parametrize(
    ("memory_used", "utilization", "pids"),
    [(3200, 5, {10, 11}), (9000, 0, {10, 11}), (100, 0, {11})],
)
def test_hotgpu_rejects_unsafe_or_missing_reservation(memory_used, utilization, pids):
    environment = {
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
        "TINY_OMNI_GPU_GATE_PARENT_PID": "10",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": "14000",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": "8000",
    }
    with pytest.raises(RuntimeError, match="handoff"):
        validate_gpu_preflight(
            memory_used_mib=memory_used,
            utilization_pct=utilization,
            total_mib=16384,
            active_pids=pids,
            own_pid=11,
            environment=environment,
        )


def test_long_unique_coverage_config_pins_the_repeat_control_and_32k_schedule():
    config = json.loads(
        Path("configs/pretrained_reuse/native_unique_coverage_long_v1.json").read_text()
    )
    control = config["comparison_control"]

    assert config["id"] == "native-unique-coverage-long-v1"
    assert config["training"]["seed"] == 17
    assert config["training"]["max_steps"] == 32768
    assert config["training"]["evaluation_steps"][-1] == 32768
    assert config["pools"].keys() == {"repeat", "fresh"}
    assert control["policy"] == "repeat"
    assert control["steps"] == 32768
    assert control["result_sha256"] == config["input_pins"][control["result"]]
    assert control["validation_ids_sha256"] == config["input_pins"][control["validation_ids"]]
