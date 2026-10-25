"""Validate an idle GPU or the expected cooperative HotGPU reservation."""

import psutil

GATE_VARIABLES = (
    "TINY_OMNI_GPU_GATE_RESERVED_MIB",
    "TINY_OMNI_GPU_GATE_PARENT_PID",
    "TINY_OMNI_GPU_GATE_FREE_MIB",
    "TINY_OMNI_GPU_GATE_TOTAL_MIB",
    "TINY_OMNI_GPU_GATE_UTILIZATION_PCT",
    "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB",
    "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB",
)


def validate_gpu_preflight(
    environment,
    *,
    active_pids,
    memory_used_mib,
    memory_total_mib,
    utilization_pct,
    parent_command,
    active_process_commands=None,
):
    """Fail closed unless GPU is idle or held by a valid HotGPU guard only."""
    supplied = {name for name in GATE_VARIABLES if name in environment}
    if not supplied:
        if active_pids or memory_used_mib > 512 or utilization_pct >= 5:
            raise RuntimeError("GPU occupied; no cooperative HotGPU reservation is active")
        return {"status": "PASS_GPU_IDLE", "free_mib": memory_total_mib - memory_used_mib}

    if supplied != set(GATE_VARIABLES):
        raise RuntimeError("incomplete HotGPU reservation metadata")
    try:
        reserved_mib = int(environment[GATE_VARIABLES[0]])
        parent_pid = int(environment[GATE_VARIABLES[1]])
        declared_free_mib = int(environment[GATE_VARIABLES[2]])
        declared_total_mib = int(environment[GATE_VARIABLES[3]])
        declared_utilization = int(environment[GATE_VARIABLES[4]])
        block_free_mib = int(environment[GATE_VARIABLES[5]])
        min_child_free_mib = int(environment[GATE_VARIABLES[6]])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid HotGPU reservation metadata") from exc

    if min(reserved_mib, parent_pid, declared_total_mib, block_free_mib, min_child_free_mib) <= 0:
        raise RuntimeError("invalid HotGPU reservation metadata")
    command = " ".join(parent_command or ()).replace("\\", "/").casefold()
    if "run_with_hotgpu_guard.py" not in command:
        raise RuntimeError("HotGPU parent is not the cooperative guard process")
    auxiliary_pids = active_pids - {parent_pid}
    process_commands = active_process_commands or {}
    unverified_pids = {
        pid
        for pid in auxiliary_pids
        if not is_safe_ui_process(*process_commands.get(pid, ("", ())))
    }
    if unverified_pids:
        raise RuntimeError("GPU has another process or the HotGPU guard is not the sole owner")
    if utilization_pct >= 5 or declared_utilization >= 5:
        raise RuntimeError("GPU utilization is not idle under the HotGPU gate")

    free_mib = memory_total_mib - memory_used_mib
    if declared_total_mib != memory_total_mib or declared_free_mib >= block_free_mib:
        raise RuntimeError("HotGPU reservation does not match the observed GPU")
    if free_mib < min_child_free_mib or declared_free_mib < min_child_free_mib:
        raise RuntimeError("HotGPU reservation leaves insufficient child free-memory")
    if free_mib >= block_free_mib or abs(free_mib - declared_free_mib) > 512:
        raise RuntimeError("GPU free-memory changed outside the HotGPU reservation")
    if memory_total_mib - declared_free_mib < reserved_mib:
        raise RuntimeError("HotGPU reservation size is inconsistent with its snapshot")

    return {
        "status": "PASS_HOTGPU_RESERVED",
        "parent_pid": parent_pid,
        "free_mib": free_mib,
        "verified_ui_pids": sorted(auxiliary_pids),
    }


def is_safe_ui_process(name, command_line):
    """Recognize only Cursor's GPU renderer; this process is not model compute."""
    return name.casefold() == "cursor.exe" and "--type=gpu-process" in " ".join(
        command_line
    ).casefold()


def process_command(pid):
    """Return a process identity suitable for the narrow UI exception above."""
    try:
        process = psutil.Process(pid)
        return process.name(), process.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "", ()


def verified_ui_pids(pids):
    """Filter known UI rendering PIDs while leaving all unknown PIDs visible."""
    return {pid for pid in pids if is_safe_ui_process(*process_command(pid))}
