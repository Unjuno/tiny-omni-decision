"""Fail-closed validation for local HotGPU reservations."""

from importlib import import_module

import pytest


def _validate_gpu_preflight(*args, **kwargs):
    try:
        module = import_module("scripts.native_gpu_gate")
    except ModuleNotFoundError:
        pytest.fail("HotGPU-aware native GPU preflight validator is missing")
    validator = getattr(module, "validate_gpu_preflight", None)
    assert callable(validator), "HotGPU-aware native GPU preflight validator is missing"
    return validator(*args, **kwargs)


def _reservation():
    return {
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "2471",
        "TINY_OMNI_GPU_GATE_PARENT_PID": "700",
        "TINY_OMNI_GPU_GATE_FREE_MIB": "13744",
        "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
        "TINY_OMNI_GPU_GATE_UTILIZATION_PCT": "0",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": "14000",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": "12288",
    }


def test_accepts_only_the_live_hotgpu_parent_reservation():
    result = _validate_gpu_preflight(
        _reservation(),
        active_pids={700},
        memory_used_mib=2640,
        memory_total_mib=16384,
        utilization_pct=0,
        parent_command=["python.exe", "scripts/run_with_hotgpu_guard.py"],
    )

    assert result == {
        "status": "PASS_HOTGPU_RESERVED",
        "parent_pid": 700,
        "free_mib": 13744,
    }


def test_accepts_idle_gpu_without_hotgpu_metadata():
    result = _validate_gpu_preflight(
        {},
        active_pids=set(),
        memory_used_mib=169,
        memory_total_mib=16384,
        utilization_pct=0,
        parent_command=None,
    )

    assert result["status"] == "PASS_GPU_IDLE"


@pytest.mark.parametrize(
    "changes,active_pids,used,utilization,parent_command,match",
    [
        ({}, {700, 16364}, 2640, 0, ["run_with_hotgpu_guard.py"], "other process"),
        ({"TINY_OMNI_GPU_GATE_FREE_MIB": "10000"}, {700}, 6384, 0,
         ["run_with_hotgpu_guard.py"], "child free-memory"),
        ({}, {700}, 2640, 8, ["run_with_hotgpu_guard.py"], "utilization"),
        ({}, {700}, 2640, 0, ["python.exe", "unrelated.py"], "guard process"),
        ({"TINY_OMNI_GPU_GATE_PARENT_PID": ""}, {700}, 2640, 0,
         ["run_with_hotgpu_guard.py"], "invalid HotGPU"),
    ],
)
def test_rejects_invalid_or_competing_gpu_gate(
    changes, active_pids, used, utilization, parent_command, match
):
    env = _reservation()
    env.update(changes)

    with pytest.raises(RuntimeError, match=match):
        _validate_gpu_preflight(
            env,
            active_pids=active_pids,
            memory_used_mib=used,
            memory_total_mib=16384,
            utilization_pct=utilization,
            parent_command=parent_command,
        )


def test_unreserved_gpu_usage_fails_closed():
    with pytest.raises(RuntimeError, match="GPU occupied"):
        _validate_gpu_preflight(
            {},
            active_pids={16364},
            memory_used_mib=15821,
            memory_total_mib=16384,
            utilization_pct=20,
            parent_command=None,
        )
