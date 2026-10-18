"""Wait for a cooperative GPU yield and reserve VRAM before launching a child."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def active_gpu_pids() -> set[int]:
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


def telemetry() -> tuple[int, int, int]:
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
    utilization, used_mib, total_mib = (
        int(value.strip()) for value in result.stdout.splitlines()[0].split(",")
    )
    return utilization, used_mib, total_mib


def reservation_size(total_mib: int, used_mib: int, reserve_mib: int, block_free_mib: int) -> int:
    free_mib = total_mib - used_mib
    return max(reserve_mib, free_mib - block_free_mib + 256)


def launch_conditions_met(
    active_pids: set[int],
    own_pids: set[int],
    utilization: int,
    free_mib: int,
    block_free_mib: int,
    min_child_free_mib: int,
) -> bool:
    return (
        not (active_pids - own_pids)
        and utilization < 5
        and free_mib < block_free_mib
        and free_mib >= min_child_free_mib
    )


def reserve_after_yield(
    torch_module,
    handoff_guard,
    deadline: float,
    reserve_mib: int,
    block_free_mib: int,
    min_child_free_mib: int,
):
    own_pid = os.getpid()
    while time.monotonic() < deadline:
        pids = active_gpu_pids() - {own_pid}
        utilization, used_mib, total_mib = telemetry()
        if pids or utilization >= 5:
            print(
                f"waiting for cooperative yield: pids={sorted(pids)}, "
                f"util={utilization}%, used={used_mib}MiB",
                flush=True,
            )
            time.sleep(2)
            continue

        amount_mib = reservation_size(total_mib, used_mib, reserve_mib, block_free_mib)
        if total_mib - amount_mib < min_child_free_mib:
            raise RuntimeError("GPU cannot satisfy both reservation and child free-memory limits")

        reservation = None
        try:
            # Reserve immediately after the first idle sample; no quiet-period
            # delay leaves the supervisor an avoidable restart window.
            reservation = torch_module.empty(
                amount_mib * 1024 * 1024, dtype=torch_module.uint8, device="cuda"
            )
            reservation.fill_(1)
            torch_module.cuda.synchronize()
        except torch_module.cuda.OutOfMemoryError:
            del reservation
            torch_module.cuda.empty_cache()
            print(
                "GPU allocation raced with another process; waiting for its cooperative yield",
                flush=True,
            )
            time.sleep(2)
            continue

        utilization_after, used_after, total_after = telemetry()
        free_after = total_after - used_after
        pids_after = active_gpu_pids() - {own_pid}
        if not launch_conditions_met(
            pids_after,
            {own_pid},
            utilization_after,
            free_after,
            block_free_mib,
            min_child_free_mib,
        ):
            del reservation
            torch_module.cuda.empty_cache()
            print(
                f"handoff raced: pids={sorted(pids_after)}, util={utilization_after}%, "
                f"free={free_after}MiB; released reservation and will wait again",
                flush=True,
            )
            time.sleep(2)
            continue
        return (
            handoff_guard,
            reservation,
            {
                "reserved_mib": amount_mib,
                "free_mib": free_after,
                "total_mib": total_after,
                "utilization_pct": utilization_after,
            },
        )
    raise TimeoutError("GPU did not yield before the deadline; child was not started")


def build_child_environment(
    base_environment: dict[str, str],
    parent_pid: int,
    slot: dict[str, int],
    block_if_free_mib: int,
    min_child_free_mib: int,
) -> dict[str, str]:
    environment = base_environment.copy()
    environment["TINY_OMNI_GPU_GATE_RESERVED_MIB"] = str(slot["reserved_mib"])
    environment["TINY_OMNI_GPU_GATE_PARENT_PID"] = str(parent_pid)
    environment["TINY_OMNI_GPU_GATE_FREE_MIB"] = str(slot["free_mib"])
    environment["TINY_OMNI_GPU_GATE_TOTAL_MIB"] = str(slot["total_mib"])
    environment["TINY_OMNI_GPU_GATE_UTILIZATION_PCT"] = str(slot["utilization_pct"])
    environment["TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB"] = str(block_if_free_mib)
    environment["TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB"] = str(min_child_free_mib)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    parser.add_argument("--reserve-mib", type=int, default=3200)
    parser.add_argument("--block-if-free-mib", type=int, default=14000)
    parser.add_argument("--min-child-free-mib", type=int, default=8000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        min(
            args.max_wait_seconds, args.reserve_mib, args.block_if_free_mib, args.min_child_free_mib
        )
        < 1
    ):
        raise ValueError("positive wait and GPU memory limits are required")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("child command is required")
    command[0] = str(Path(command[0]).resolve(strict=True))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; child will not start")
    torch.cuda.init()
    handoff_guard = torch.empty(1, dtype=torch.uint8, device="cuda")
    handoff_guard.fill_(1)
    torch.cuda.synchronize()

    try:
        handoff_guard, reservation, slot = reserve_after_yield(
            torch,
            handoff_guard,
            time.monotonic() + args.max_wait_seconds,
            args.reserve_mib,
            args.block_if_free_mib,
            args.min_child_free_mib,
        )
        environment = build_child_environment(
            os.environ.copy(),
            os.getpid(),
            slot,
            args.block_if_free_mib,
            args.min_child_free_mib,
        )
        print(
            f"HotUGPU reservation active: {slot['reserved_mib']}MiB reserved, "
            f"free={slot['free_mib']}MiB; child starting under the free-memory gate",
            flush=True,
        )
        completed = subprocess.run(command, check=False, env=environment)
    finally:
        if "reservation" in locals():
            del reservation
        del handoff_guard
        torch.cuda.empty_cache()
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
