"""Reserve GPU memory before launching a child under the native HotGPU gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from scripts.native_gpu_gate import verified_ui_pids


def active_gpu_pids():
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


def telemetry():
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


def reservation_size(total_mib, used_mib, reserve_mib, block_free_mib):
    free_mib = total_mib - used_mib
    return max(reserve_mib, free_mib - block_free_mib + 256)


def launch_conditions_met(
    active_pids,
    own_pids,
    safe_ui_processes,
    utilization,
    free_mib,
    block_free_mib,
    min_child_free_mib,
):
    return (
        not (active_pids - own_pids - safe_ui_processes)
        and utilization < 5
        and free_mib < block_free_mib
        and free_mib >= min_child_free_mib
    )


def build_child_environment(base_environment, parent_pid, slot, block_free_mib, min_child_free_mib):
    environment = base_environment.copy()
    environment.update(
        {
            "TINY_OMNI_GPU_GATE_RESERVED_MIB": str(slot["reserved_mib"]),
            "TINY_OMNI_GPU_GATE_PARENT_PID": str(parent_pid),
            "TINY_OMNI_GPU_GATE_FREE_MIB": str(slot["free_mib"]),
            "TINY_OMNI_GPU_GATE_TOTAL_MIB": str(slot["total_mib"]),
            "TINY_OMNI_GPU_GATE_UTILIZATION_PCT": str(slot["utilization_pct"]),
            "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": str(block_free_mib),
            "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": str(min_child_free_mib),
        }
    )
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def reserve_after_yield(
    torch_module,
    handoff_guard,
    deadline,
    reserve_mib,
    block_free_mib,
    min_child_free_mib,
):
    own_pid = os.getpid()
    while time.monotonic() < deadline:
        pids = active_gpu_pids()
        safe_ui = verified_ui_pids(pids - {own_pid})
        competitors = pids - {own_pid} - safe_ui
        utilization, used_mib, total_mib = telemetry()
        if competitors or utilization >= 5:
            print(
                f"waiting for cooperative yield: pids={sorted(competitors)}, "
                f"util={utilization}%, used={used_mib}MiB",
                flush=True,
            )
            time.sleep(2)
            continue

        amount_mib = reservation_size(total_mib, used_mib, reserve_mib, block_free_mib)
        if total_mib - amount_mib < min_child_free_mib:
            raise RuntimeError("GPU cannot satisfy reservation and child free-memory limits")

        reservation = None
        try:
            reservation = torch_module.empty(
                amount_mib * 1024 * 1024, dtype=torch_module.uint8, device="cuda"
            )
            reservation.fill_(1)
            torch_module.cuda.synchronize()
        except torch_module.cuda.OutOfMemoryError:
            del reservation
            torch_module.cuda.empty_cache()
            print("GPU allocation raced; waiting for cooperative yield", flush=True)
            time.sleep(2)
            continue

        utilization_after, used_after, total_after = telemetry()
        pids_after = active_gpu_pids()
        safe_ui_after = verified_ui_pids(pids_after - {own_pid})
        free_after = total_after - used_after
        if not launch_conditions_met(
            pids_after,
            {own_pid},
            safe_ui_after,
            utilization_after,
            free_after,
            block_free_mib,
            min_child_free_mib,
        ):
            del reservation
            torch_module.cuda.empty_cache()
            print("GPU handoff raced; reservation released", flush=True)
            time.sleep(2)
            continue

        return reservation, {
            "reserved_mib": amount_mib,
            "free_mib": free_after,
            "total_mib": total_after,
            "utilization_pct": utilization_after,
        }
    raise TimeoutError("GPU did not yield before the deadline; child was not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    parser.add_argument("--reserve-mib", type=int, default=1)
    parser.add_argument("--block-if-free-mib", type=int, default=14000)
    parser.add_argument("--min-child-free-mib", type=int, default=12288)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (
        min(
            args.max_wait_seconds,
            args.reserve_mib,
            args.block_if_free_mib,
            args.min_child_free_mib,
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
        reservation, slot = reserve_after_yield(
            torch,
            handoff_guard,
            time.monotonic() + args.max_wait_seconds,
            args.reserve_mib,
            args.block_if_free_mib,
            args.min_child_free_mib,
        )
        print(
            f"HotGPU reservation active: {slot['reserved_mib']} MiB; "
            f"free={slot['free_mib']} MiB; launching protected sequence",
            flush=True,
        )
        environment = build_child_environment(
            os.environ.copy(),
            os.getpid(),
            slot,
            args.block_if_free_mib,
            args.min_child_free_mib,
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
