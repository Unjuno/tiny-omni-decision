"""Score the frozen target-free AV replacement inventory for E0G0 and E0G1."""
# ruff: noqa: E402 - pin this worktree's src/ before importing project modules.

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_SOURCE = str(REPO / "src")
if LOCAL_SOURCE in sys.path:
    sys.path.remove(LOCAL_SOURCE)
sys.path.insert(0, LOCAL_SOURCE)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import psutil
import torch

from scripts.greatest_hits_matched import digest
from scripts.greatest_hits_matched_inputs import (
    AUDIO,
    ROOT,
    TASKS,
    VIDEO,
    Pins,
    read,
    receipt,
)
from scripts.greatest_hits_matched_runtime import create_model
from scripts.greatest_hits_replacement import build_inventory
from scripts.greatest_hits_replacement_scoring import score
from scripts.greatest_hits_transport_store import ExplicitEventStore
from scripts.native_video_cache import sha
from scripts.train_greatest_hits_matched import write

INVENTORY = ROOT / "greatest-hits-replacement-inventory-01"
INVENTORY_SHA256 = "f12690080072e7cb0c60e0528394367114afc296f8c9cbe3d281425937687c62"
AUDIT = ROOT / "greatest-hits-matched-input-audit-01"
AUDIT_SHA256 = "69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad"
RUNS = {
    "E0G0": ROOT / "greatest-hits-matched-postnorm-train-01",
    "E0G1": ROOT / "greatest-hits-matched-postnorm-train-E0G1-01",
}
RUN_RESULT_SHA256 = {
    "E0G0": "2b0c300ba650e3317000a579154c4ad05d1f8320c4dda2740b2a09b21d9a0825",
    "E0G1": "a8617ac91db29c269b6adbc6e2ea149b9a2755b91541748d1c0c367ac9ed4520",
}
CHECKPOINT_SHA256 = {
    "E0G0": "d08120c77ec766427ca5d789629a1f885265f295842784596165c4858ba09b12",
    "E0G1": "9df53a55e610f4e473dc5f56528edd86610a72683c8f6c40145e9d87d899d87d",
}
OUTPUT_PREFIX = "greatest-hits-postnorm-replacements-"
GPU_PROCESS_POLL_SECONDS = 2.0


def active_gpu_pids():
    report = subprocess.check_output(["nvidia-smi"], text=True)
    return {
        int(match.group(1))
        for line in report.splitlines()
        if (match := re.match(r"\|\s+\d+\s+N/A\s+N/A\s+(\d+)\s+C\s+", line))
    }


def refresh_gpu_pids(now, checked_at, cached):
    """Reuse the last GPU-process snapshot for a short bounded interval."""
    if now - checked_at >= GPU_PROCESS_POLL_SECONDS:
        return active_gpu_pids(), now
    return cached, checked_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(RUNS), required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.attempt < 1 or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO
    ).strip():
        raise ValueError("positive resource limit and clean committed source required")
    output = ROOT / (OUTPUT_PREFIX + args.arm + f"-{args.attempt:02d}")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    pins, sources = Pins(), Pins()
    result = dict(
        status="RUNNING",
        arm=args.arm,
        pid=os.getpid(),
        command=[sys.executable, *sys.argv],
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        started_utc=datetime.now(UTC).isoformat(),
        optimizer_updates=0,
        encoder_updates=0,
        model_forwards=0,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        cloud_cost=0,
        checkpoint_selection=False,
        max_seconds=args.max_seconds,
    )
    last_check = 0.0
    last_gpu_check = float("-inf")
    cached_gpu_pids = set()

    def guard():
        nonlocal last_check, last_gpu_check, cached_gpu_pids
        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        result.update(
            peak_working_set_bytes=psutil.Process().memory_info().peak_wset,
            peak_vram_bytes=peak_vram,
            model_forwards=result.get("model_forwards", 0),
        )
        if (
            elapsed > args.max_seconds
            or result["peak_working_set_bytes"] > 10 * 2**30
            or peak_vram > 6 * 2**30
            or psutil.disk_usage(str(output)).free < 16 * 2**30
        ):
            raise RuntimeError("resource guard; preserve incomplete diagnostic")
        cached_gpu_pids, last_gpu_check = refresh_gpu_pids(
            elapsed, last_gpu_check, cached_gpu_pids
        )
        others = cached_gpu_pids - {os.getpid()}
        if others:
            raise RuntimeError(f"GPU contention from other compute process(es): {sorted(others)}")
        if elapsed - last_check >= 15:
            pins.check()
            sources.check()
            last_check = elapsed

    try:
        write(output / "result.json", result)
        gpu_pids = active_gpu_pids()
        result["preflight_gpu_processes"] = sorted(gpu_pids)
        write(output / "result.json", result)
        if gpu_pids:
            raise RuntimeError("GPU occupied; scoring not started")
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0"):
            if name:
                sources.add(REPO / name)
        run_folder = RUNS[args.arm]
        run = receipt(pins, run_folder, RUN_RESULT_SHA256[args.arm])
        if (
            run["status"] != "PASS_MATCHED_AV_TRAIN"
            or run["arm"] != args.arm
            or run["step"] != 32768
            or run["best_final_reload_verified"] is not True
            or run["sealed_audit_loaded"] is not False
            or run["config"].get("candidate_output_norm") is not True
        ):
            raise ValueError("completed selected postnorm run required")
        # Training was pinned to an earlier commit; verify every Python input against it.
        source_root = Path(run["command"][1]).resolve().parents[1]
        for path, expected in run["source_pins"].items():
            relative = Path(path).resolve().relative_to(source_root)
            sources.add(REPO / relative, expected)
        admitted = receipt(pins, AUDIT, AUDIT_SHA256)
        proof = read(AUDIT / "admission.json")
        if (
            admitted["status"] != "PASS_MATCHED_AV_INPUT_AUDIT"
            or proof["input_digest"] != digest(proof["input_pins"])
            or proof["input_digest"] != admitted["input_digest"]
            or proof["input_digest"] != run["input_digest"]
        ):
            raise ValueError("the exact training-admitted frozen inputs are required")
        for path, expected in proof["input_pins"].items():
            pins.add(path, expected)
        inventory_receipt = receipt(pins, INVENTORY, INVENTORY_SHA256)
        if inventory_receipt["status"] != "PASS_FROZEN_AV_REPLACEMENT_INVENTORY":
            raise ValueError("original frozen label-blind inventory required")
        inventory = read(INVENTORY / "inventory.json")
        rows = [
            json.loads(line)
            for line in (TASKS / "validation.jsonl").read_text(encoding="utf8").splitlines()
        ]
        if len(rows) != 8875 or inventory != build_inventory(rows):
            raise ValueError("complete original validation/inventory required")
        best = read(run_folder / "best.json")
        checkpoint_path = run_folder / "best.pt"
        pins.add(checkpoint_path, CHECKPOINT_SHA256[args.arm])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            best["step"] != run["best_step"]
            or sha(checkpoint_path) != best["checkpoint_sha256"]
            or checkpoint["step"] != run["best_step"]
            or checkpoint["sampler_sha256"] != run["config"]["plan_sha256"]
            or checkpoint["config_sha256"] != run["config_sha256"]
        ):
            raise ValueError("selected checkpoint/config/sampler binding differs")
        saved_rows = read(run_folder / "evaluation-rows.json")["validation-joint"]
        expected_scores = np.load(
            run_folder / f"step-{run['best_step']:05}-validation-joint.npy",
            allow_pickle=False,
        )
        if expected_scores.shape != (8875, 40) or expected_scores.dtype != np.float32:
            raise ValueError("selected natural validation reference shape/dtype differs")
        by_id = {row["id"]: row for row in rows}
        if {row["id"] for row in saved_rows} != set(by_id):
            raise ValueError("saved natural validation reference IDs differ")
        natural = {
            row["id"]: expected_scores[i, : len(by_id[row["id"]]["options"])].copy()
            for i, row in enumerate(saved_rows)
        }
        gpu_pids = active_gpu_pids()
        if gpu_pids:
            raise RuntimeError("GPU occupied; scoring not started")
        if not torch.cuda.is_available():
            raise RuntimeError("existing local CUDA device required")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = create_model(args.arm, candidate_output_norm=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        if sum(param.numel() for param in model.parameters()) != 1732804:
            raise ValueError("registered postnorm parameter count differs")
        model.cuda()
        store = ExplicitEventStore(
            proof["binding"],
            dict(audio=AUDIO, video=VIDEO),
            dict(
                audio=read(AUDIO / "result.json")["artifacts"],
                video=read(VIDEO / "result.json")["artifacts"],
            ),
            device="cuda",
        )
        result.update(
            selected_step=run["best_step"],
            checkpoint_sha256=sha(checkpoint_path),
            input_digest=run["input_digest"],
            inventory_pairs_sha256=inventory["pairs_sha256"],
            python=sys.version,
            versions={
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "psutil")
            },
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            preflight_gpu_processes=sorted(gpu_pids),
        )
        with (output / "predictions.jsonl").open("x", encoding="utf8") as stream:
            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                result["model_forwards"] += 1
                if result["model_forwards"] % 500 == 0:
                    stream.flush()
                    write(output / "result.json", result)
                    print("forwards", result["model_forwards"], "/44375", flush=True)

            summary = score(
                model,
                store,
                rows,
                inventory,
                emit=emit,
                expected_natural=natural,
                guard=guard,
            )
        if summary["model_forwards"] != 44375 or summary["self_replacements_exact"] != 17750:
            raise ValueError("full five-view coverage or exact self-replay failed")
        write(output / "sensitivity.json", summary)
        store.verify_files()
        pins.verify()
        sources.verify()
        guard()
        result.update(
            status="PASS_POSTNORM_AV_REPLACEMENTS",
            self_replacements_exact=17750,
            counterfactual_truth_verified=False,
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started,
            finished_utc=datetime.now(UTC).isoformat(),
            input_pins=pins.hashes,
            source_pins=sources.hashes,
        )
        write(output / "result.json", result)
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
