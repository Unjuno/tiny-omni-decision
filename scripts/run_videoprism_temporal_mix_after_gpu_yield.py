"""Request cooperative GPU yield and reserve VRAM against Orbitune auto-launch."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def _active_gpu_pids() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip() and line.strip().casefold() != "no running processes found"
    }


def _telemetry() -> tuple[float, int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utilization, used, total = (
        int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
    )
    return float(utilization), used, total


def _create_handoff_guard(torch_module):
    """Expose a CUDA PID before Orbitune's yielding trainer exits."""
    torch_module.cuda.init()
    guard = torch_module.empty(1, dtype=torch_module.uint8, device="cuda")
    guard.fill_(1)
    torch_module.cuda.synchronize()
    return guard


def _wait_for_idle(max_wait_seconds: int, quiet_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + max_wait_seconds
    own_pid = os.getpid()
    idle_since = None
    while time.monotonic() < deadline:
        pids = _active_gpu_pids() - {own_pid}
        utilization, used_mib, _ = _telemetry()
        if pids or utilization >= 5 or used_mib >= 2200:
            idle_since = None
            print(
                f"waiting for cooperative yield: pids={sorted(pids)}, "
                f"util={utilization:.0f}%, memory={used_mib}MiB",
                flush=True,
            )
        else:
            idle_since = idle_since or time.monotonic()
            stable = time.monotonic() - idle_since
            print(f"GPU idle for {stable:.1f}/{quiet_seconds:.0f}s", flush=True)
            if stable >= quiet_seconds:
                return
        time.sleep(2)
    raise TimeoutError("GPU did not yield; no VRAM reservation or child job was started")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    parser.add_argument("--reserve-mib", type=int, default=3200)
    parser.add_argument("--block-if-free-mib", type=int, default=14000)
    parser.add_argument("--min-child-free-mib", type=int, default=8000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        args.max_wait_seconds < 1
        or args.reserve_mib < 1
        or args.block_if_free_mib < 1
        or args.min_child_free_mib < 1
        or not args.command
    ):
        raise ValueError("positive limits and a child command are required")
    command = args.command[1:] if args.command[0] == "--" else args.command
    executable = Path(command[0]).resolve(strict=True)
    command[0] = str(executable)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; child GPU job will not start")
    handoff_guard = _create_handoff_guard(torch)
    process_id = os.getpid()
    # Keep this tiny CUDA allocation alive while Orbitune's trainer cooperatively
    # exits, so its supervisor sees our PID before deciding whether to relaunch.
    _wait_for_idle(args.max_wait_seconds)
    _, _, total_mib = _telemetry()
    reserve_mib = max(args.reserve_mib, total_mib - args.block_if_free_mib + 256)
    if total_mib - reserve_mib < args.min_child_free_mib:
        raise RuntimeError("VRAM block and child free-memory limits cannot both be met")
    reservation = torch.empty(reserve_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
    reservation.fill_(1)
    torch.cuda.synchronize()
    utilization, used_mib, total_mib = _telemetry()
    free_mib = total_mib - used_mib
    if free_mib >= args.block_if_free_mib or free_mib < args.min_child_free_mib:
        del reservation
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"VRAM reservation preflight failed: free={free_mib}MiB, "
            f"required <{args.block_if_free_mib}MiB and >={args.min_child_free_mib}MiB"
        )
    other_pids = _active_gpu_pids() - {process_id}
    if other_pids or utilization >= 5:
        del reservation
        torch.cuda.empty_cache()
        raise RuntimeError("another GPU process appeared during the reservation handoff")
    child_environment = os.environ.copy()
    child_environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    child_environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    child_environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    child_environment["TINY_OMNI_GPU_GATE_RESERVED_MIB"] = str(reserve_mib)
    child_environment["TINY_OMNI_GPU_GATE_FREE_MIB"] = str(free_mib)
    child_environment["TINY_OMNI_GPU_GATE_TOTAL_MIB"] = str(total_mib)
    child_environment["TINY_OMNI_GPU_GATE_UTILIZATION_PCT"] = str(int(utilization))
    child_environment["TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB"] = str(args.block_if_free_mib)
    child_environment["TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB"] = str(args.min_child_free_mib)
    print(
        f"HotUGPU reservation active: {reserve_mib}MiB reserved, free={free_mib}MiB; "
        "launching child while Orbitune's configured free-memory gate is closed",
        flush=True,
    )
    try:
        completed = subprocess.run(command, check=False, env=child_environment)
    finally:
        del reservation
        del handoff_guard
        torch.cuda.empty_cache()
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
