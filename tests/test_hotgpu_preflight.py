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
