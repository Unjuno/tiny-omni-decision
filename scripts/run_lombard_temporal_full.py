"""Separate full-fit controller; requires frozen smoke and evaluator contract."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.clap_progress_io import save_progress
from scripts.lombard_readout_data import read_json, sha
from scripts.lombard_temporal_full import admit_complete_smoke
from scripts.robomnist_audio_cue import recheck
from scripts.train_lombard_temporal import (
    ARMS,
    CONFIG,
    bound_sources,
    fit_arm,
    guard,
    source_scope,
)

EXTRA = [
    "scripts/run_lombard_temporal_full.py",
    "scripts/lombard_temporal_full.py",
    "scripts/evaluate_lombard_temporal.py",
    "tests/test_lombard_temporal_full.py",
    "scripts/report_lombard_grid_readouts.py",
    "scripts/starss23_readout_common.py",
    "docs/pretrained_reuse/R2_LOMBARD_TEMPORAL_FULL_PROTOCOL_20261015.md",
]


def full_pins():
    return {str(Path(p).resolve()): sha(Path(p)) for p in EXTRA}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-reference", type=Path, required=True)
    parser.add_argument("--smoke-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    report = dict(
        status="RUNNING",
        arms={},
        smoke=False,
        device="cpu",
        gpu_used=False,
        encoders_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        started_utc=datetime.now(UTC).isoformat(),
        command=subprocess.list2cmdline(
            [sys.executable, "-m", "scripts.run_lombard_temporal_full", *sys.argv[1:]]
        ),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    try:
        guard(args.output, tick)
        pins, extra = bound_sources(), full_pins()
        recheck(pins)
        recheck(extra)
        smoke = read_json(args.smoke_reference / "result.json", args.smoke_sha256)
        admit_complete_smoke(smoke, pins)
        if sha(args.smoke_reference / "effective-config.json") != smoke["effective_config_sha256"]:
            raise ValueError("smoke effective config changed")
        for arm in ARMS:
            recheck(
                {
                    str((args.smoke_reference / arm / k).resolve()): v
                    for k, v in smoke["arms"][arm]["artifact_hashes"].items()
                }
            )
        config = read_json(CONFIG, pins[str(CONFIG.resolve())])
        if config != smoke["config"]:
            raise ValueError("fixed smoke config differs")
        report.update(
            validated_hashes=pins,
            full_execution_hashes=extra,
            config=config,
            smoke_sha256=args.smoke_sha256,
            smoke_reference=str(args.smoke_reference.resolve()),
            python=smoke["python"],
            torch=torch.__version__,
        )
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        raw, framed, targets, rows, provenance = source_scope(False)
        if [len(rows[s]) for s in rows] != [4080, 398, 889]:
            raise ValueError("all original full split counts required")
        report["provenance"] = provenance
        (args.output / "effective-config.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        report["effective_config_sha256"] = sha(args.output / "effective-config.json")
        for arm in ARMS:
            report["arms"][arm] = fit_arm(
                raw, framed, targets, rows, config, arm, args.output / arm, args.output, tick, False
            )
            save_progress(args.output / "progress.json", report)
        if any(r["status"] != "PASS_ARM_EXECUTION" for r in report["arms"].values()):
            raise ValueError("all five fits must complete")
        for key in ["initial_tensor_sha256", "sampler_sha256"]:
            if len({report["arms"][a][key] for a in ARMS}) != 1:
                raise ValueError("full initialization/sample order differs")
        if (
            report["arms"][ARMS[0]]["initial_tensor_sha256"]
            != smoke["arms"][ARMS[0]]["initial_tensor_sha256"]
        ):
            raise ValueError("full head differs from admitted smoke initialization")
        recheck(pins)
        recheck(extra)
        if source_scope(False)[4] != provenance:
            raise ValueError("source authority changed")
        if sha(args.output / "effective-config.json") != report["effective_config_sha256"]:
            raise ValueError("effective config changed")
        if torch.cuda.is_initialized():
            raise ValueError("CPU controller initialized CUDA")
        report.update(status="PASS_FIVE_ARM_TRAINING", independent_evaluation_pending=True)
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        try:
            report.update(**guard(args.output, tick))
        except Exception as error:
            report.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        report["ended_utc"] = datetime.now(UTC).isoformat()
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if report["status"] != "PASS_FIVE_ARM_TRAINING":
        raise RuntimeError("full training failed finalization")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
