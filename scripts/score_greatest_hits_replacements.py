"""One immutable selected-arm diagnostic after all five matched selections freeze."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import psutil
import torch

from scripts.greatest_hits_matched import digest
from scripts.greatest_hits_matched_inputs import AUDIO, ROOT, TASKS, VIDEO, Pins, read, receipt
from scripts.greatest_hits_matched_runtime import create_model
from scripts.greatest_hits_replacement_scoring import score
from scripts.greatest_hits_selected_admission import (
    ARMS,
    natural_reference,
    require_selected,
    require_selection_set,
)
from scripts.greatest_hits_transport_store import ExplicitEventStore
from scripts.native_video_cache import sha
from scripts.train_native_long_budget import write

REPO = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "greatest-hits-replacement-inventory-01"
INVENTORY_SHA = "f12690080072e7cb0c60e0528394367114afc296f8c9cbe3d281425937687c62"
PROTOCOL = REPO / "docs/pretrained_reuse/R2_GREATEST_HITS_INPUT_REPLACEMENT_PROTOCOL_20261017.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-set", type=Path, required=True)
    parser.add_argument("--selection-set-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError("explicit positive resource allowance required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed diagnostic source required")
    # Fixed per-arm destinations prevent accidental repeat evaluation/new output retries.
    output = ROOT / ("greatest-hits-selected-replacements-" + args.arm + "-01")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    sources, pins = Pins(), Pins()
    result = dict(
        status="RUNNING",
        arm=args.arm,
        pid=os.getpid(),
        command=[sys.executable, *sys.argv],
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
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

    def guard():
        nonlocal last_check
        elapsed = time.perf_counter() - started
        peak = psutil.Process().memory_info().peak_wset
        vram = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        result.update(peak_working_set_bytes=peak, peak_vram_bytes=vram)
        if (
            elapsed > args.max_seconds
            or peak > 10 * 2**30
            or vram > 6 * 2**30
            or psutil.disk_usage(output).free < 16 * 2**30
        ):
            raise RuntimeError("resource guard; preserve incomplete diagnostic")
        if elapsed - last_check > 15:
            sources.check()
            pins.check()
            last_check = elapsed

    try:
        write(output / "result.json", result)
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0"):
            if name:
                sources.add(REPO / name)
        sources.add(PROTOCOL)
        pins.add(args.selection_set, args.selection_set_sha256)
        selections = read(args.selection_set)
        if set(selections) != set(ARMS):
            raise ValueError("exact five-arm selection set required")
        runs = {}
        for arm, entry in selections.items():
            folder = Path(entry["path"])
            pins.add(folder / "result.json", entry["result_sha256"])
            runs[arm] = read(folder / "result.json")
        require_selection_set(runs)
        run = runs[args.arm]
        folder = Path(selections[args.arm]["path"])
        receipt(pins, folder, selections[args.arm]["result_sha256"])
        # Verify shared executed Python and training protocol against this checkout.
        source_root = Path(run["command"][1]).resolve().parents[1]
        for path, expected in run["source_pins"].items():
            relative = Path(path).resolve().relative_to(source_root)
            sources.add(REPO / relative, expected)
        audit = receipt(pins, args.audit, run["audit_sha256"])
        proof = read(args.audit / "admission.json")
        if (
            audit["status"] != "PASS_MATCHED_AV_INPUT_AUDIT"
            or proof["input_digest"] != digest(proof["input_pins"])
            or proof["input_digest"] != audit["input_digest"]
            or proof["input_digest"] != run["input_digest"]
        ):
            raise ValueError("identical admitted inputs required")
        for path, expected in proof["input_pins"].items():
            pins.add(path, expected)
            guard()
        inventory_receipt = receipt(pins, INVENTORY, INVENTORY_SHA)
        if inventory_receipt["status"] != "PASS_FROZEN_AV_REPLACEMENT_INVENTORY":
            raise ValueError("original frozen inventory required")
        inventory = read(INVENTORY / "inventory.json")
        rows = [
            json.loads(line)
            for line in (TASKS / "validation.jsonl").read_text(encoding="utf8").splitlines()
        ]
        if len(rows) != 8875:
            raise ValueError("all8875 frozen validation questions required")
        saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        require_selected(run, read(folder / "best.json"), saved, sha(folder / "best.pt"))
        natural = natural_reference(
            rows,
            read(folder / "evaluation-rows.json")["validation-joint"],
            np.load(
                folder / f"step-{run['best_step']:05}-validation-joint.npy", allow_pickle=False
            ),
        )
        # Check occupancy immediately before allocating model/feature tensors on CUDA.
        usage = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        memory, utilization = [int(v.strip()) for v in usage.split(",")]
        if memory > 512 or utilization > 5:
            raise RuntimeError("GPU occupied; diagnostic not started")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available():
            raise RuntimeError("existing local CUDA required")
        versions = {p: importlib.metadata.version(p) for p in ("torch", "numpy", "psutil")}
        backend = dict(
            flash=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),
            cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
            selection="PyTorch automatic",
        )
        if versions != run["versions"] or backend != run["attention_backend_policy"]:
            raise ValueError("selected-run runtime/backend differs")
        torch.cuda.reset_peak_memory_stats()
        model = create_model(args.arm)
        model.load_state_dict(saved["model"], strict=True)
        count = sum(p.numel() for p in model.parameters())
        if count != (889345 if args.arm == "C0" else 1732420):
            raise ValueError("registered parameter count differs")
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
            checkpoint_sha256=sha(folder / "best.pt"),
            selection_set_sha256=args.selection_set_sha256,
            input_digest=run["input_digest"],
            versions=versions,
            backend=backend,
            python=sys.version,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            encoder_inclusive_parameters=195365529 + count,
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
                model, store, rows, inventory, emit=emit, expected_natural=natural, guard=guard
            )
        if summary["model_forwards"] != 44375 or summary["self_replacements_exact"] != 17750:
            raise ValueError("full diagnostic coverage differs")
        write(output / "sensitivity.json", summary)
        store.verify_files()
        pins.verify()
        sources.verify()
        guard()
        result.update(
            status="PASS_SELECTED_AV_REPLACEMENTS",
            self_replacements_exact=17750,
            counterfactual_truth_verified=False,
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - started,
            input_pins=pins.hashes,
            source_pins=sources.hashes,
            artifacts={
                p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "result.json"
            },
        )
        write(output / "result.json", result)
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
