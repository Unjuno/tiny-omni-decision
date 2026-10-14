"""Separate-process exact best/final replay against original saved epoch outputs."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from scripts.audit_correspondence_validation import sha, write
from scripts.extract_spatial_context_corpus import load_plans
from scripts.spatial_readout_inputs import SPATIAL_RESULT_PIN, load_spatial_bank
from scripts.train_role_supervised_qa import load, verify_saved_epoch
from scripts.train_spatial_trajectory_readout import predict, process_peak_bytes, require_limits
from scripts.train_trajectory_readout import ROOT
from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("replay report already exists")
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        training=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        cases=[],
        source_run=str(args.run),
    )
    try:
        run_path, config_path = args.run / "result.json", args.run / "config.json"
        run, config = json.loads(run_path.read_text()), json.loads(config_path.read_text())
        if run["status"] != "PASS_FIXED_SPATIAL_APPEARANCE_COMPARISON_ONLY":
            raise ValueError("full completed training run required")
        if set(run["arms"]) != {"initial_broadcast", "temporal_appearance"}:
            raise ValueError("exactly two declared arms required")
        if sha(config_path) != run["config_sha256"]:
            raise ValueError("effective config changed")
        for path, pin in config["code_hashes"].items():
            if sha(Path(path)) != pin:
                raise ValueError("original executable code changed")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            raise MemoryError("GPU must be free for isolated replay")
        _, plans = load_plans()
        spatial, _ = load_spatial_bank(
            ROOT / "spatial-context-corpus-01/result.json", plans, SPATIAL_RESULT_PIN
        )
        bank, data, _, _ = load()
        limits = (start, dict(max_seconds=300, peak_cap_bytes=4 * 1024**3))
        for arm, metadata in run["arms"].items():
            for kind, epoch in (("best", metadata["best_epoch"]), ("final", 50)):
                path = args.run / arm / f"{kind}.pt"
                if sha(path) != metadata["hashes"][path.name]:
                    raise ValueError("checkpoint changed")
                original = args.run / arm / f"epoch-{epoch:02d}-predictions.npz"
                if sha(original) != metadata["hashes"][original.name]:
                    raise ValueError("original epoch outputs changed")
                model = SpatialTrajectoryReadout().cuda().eval()
                model.load_state_dict(torch.load(path, weights_only=True), strict=True)
                predictions = {
                    s: predict(model, bank, spatial, data[s], arm, "cuda", limits=limits)
                    for s in ("train", "valid")
                }
                verify_saved_epoch(original, predictions)
                report["cases"].append(
                    dict(
                        arm=arm,
                        kind=kind,
                        epoch=epoch,
                        train_questions=198,
                        validation_questions=514,
                        qa_cover_logp_bitwise=True,
                        weights_sha256=sha(path),
                    )
                )
                del model
                torch.cuda.empty_cache()
        for path, pin in config["input_hashes"].items():
            require_limits(*limits)
            if sha(Path(path)) != pin:
                raise ValueError("protected input changed")
        require_limits(*limits)
        if len(report["cases"]) != 4:
            raise ValueError("all four best/final checkpoint cases required")
        report.update(
            status="PASS_FRESH_PROCESS_ALL4_CHECKPOINT_REPLAY",
            result_sha256=sha(run_path),
            config_sha256=sha(config_path),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_process_working_set_bytes=process_peak_bytes(),
        )
    except Exception as error:
        report.update(status="FAIL_INCOMPLETE", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write(args.output, report)


if __name__ == "__main__":
    main()
