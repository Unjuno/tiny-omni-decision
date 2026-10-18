"""Run the pinned video comparison only after other GPU users yield cooperatively."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from scripts.train_textfree_video_program_order import run


def active_gpu_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = set()
    for line in result.stdout.splitlines():
        value = line.strip()
        if value and value.casefold() != "no running processes found":
            pids.add(int(value))
    return pids


def gpu_telemetry() -> tuple[float, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = result.stdout.strip().splitlines()[0]
    utilization, memory_mib = (value.strip() for value in first.split(",", maxsplit=1))
    return float(utilization), int(memory_mib)


def wait_for_yield(max_wait_seconds: int = 300) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a GPU run")
    process_id = os.getpid()
    torch.cuda.init()
    reservation = torch.empty((1,), dtype=torch.uint8, device="cuda")
    deadline = time.monotonic() + max_wait_seconds
    idle_since: float | None = None
    print(
        f"CUDA reservation PID={process_id}; waiting for cooperative yield up to "
        f"{max_wait_seconds}s",
        flush=True,
    )
    while time.monotonic() < deadline:
        other_pids = active_gpu_pids() - {process_id}
        utilization, memory_mib = gpu_telemetry()
        if other_pids or utilization >= 5.0 or memory_mib >= 2200:
            idle_since = None
            print(
                "waiting for GPU to clear: "
                f"processes={sorted(other_pids)}, utilization={utilization:.0f}%, "
                f"memory={memory_mib}MiB",
                flush=True,
            )
        else:
            idle_since = idle_since or time.monotonic()
            stable = time.monotonic() - idle_since
            print(f"GPU clear except reservation; stable={stable:.1f}s", flush=True)
            if stable >= 5.0:
                return {
                    "reservation": reservation,
                    "stable_idle_seconds": stable,
                    "idle_gpu_utilization_percent": utilization,
                    "idle_gpu_memory_used_mib": memory_mib,
                }
        time.sleep(2.0)
    del reservation
    raise TimeoutError("no cooperative GPU yield observed; training was not started")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wait-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.max_wait_seconds < 1:
        raise ValueError("max-wait-seconds must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.output_dir}")
    ready = wait_for_yield(args.max_wait_seconds)
    print("cooperative yield observed; starting fixed-config fit", flush=True)
    report = run(args.config, args.output_dir, "cuda")
    torch.cuda.synchronize()
    print(json.dumps(report, indent=2, sort_keys=True))
    del ready


if __name__ == "__main__":
    main()
