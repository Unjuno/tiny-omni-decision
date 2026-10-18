"""Validate an idle-GPU launch or a cooperative HotUGPU reservation handoff."""

from __future__ import annotations

from collections.abc import Mapping


def validate_gpu_preflight(
    *,
    memory_used_mib: int,
    utilization_pct: int,
    total_mib: int,
    active_pids: set[int],
    own_pid: int,
    environment: Mapping[str, str],
) -> dict[str, int | str]:
    """Fail closed unless the GPU is idle or reserved by this run's HotUGPU parent."""
    reserved = environment.get("TINY_OMNI_GPU_GATE_RESERVED_MIB")
    if reserved is None:
        if memory_used_mib > 512 or utilization_pct > 5 or active_pids - {own_pid}:
            raise RuntimeError("GPU occupied; do not compete")
        return {
            "mode": "unreserved_idle",
            "memory_used_mib": memory_used_mib,
            "utilization_pct": utilization_pct,
            "active_compute_pids": 0,
        }

    required = (
        "TINY_OMNI_GPU_GATE_PARENT_PID",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB",
    )
    if any(key not in environment for key in required):
        raise RuntimeError("incomplete HotUGPU reservation metadata")
    parent_pid = int(environment["TINY_OMNI_GPU_GATE_PARENT_PID"])
    reserved_mib = int(reserved)
    block_free_mib = int(environment["TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB"])
    min_child_free_mib = int(environment["TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB"])
    free_mib = total_mib - memory_used_mib
    external_pids = active_pids - {own_pid, parent_pid}
    if (
        parent_pid not in active_pids
        or external_pids
        or reserved_mib <= 0
        or utilization_pct >= 5
        or not min_child_free_mib <= free_mib < block_free_mib
    ):
        raise RuntimeError(
            "HotUGPU reservation handoff is absent, raced, or outside the free-memory gate"
        )
    return {
        "mode": "hotgpu_reserved",
        "parent_pid": parent_pid,
        "reserved_mib": reserved_mib,
        "memory_used_mib": memory_used_mib,
        "free_mib": free_mib,
        "utilization_pct": utilization_pct,
        "active_compute_pids": len(active_pids),
    }
