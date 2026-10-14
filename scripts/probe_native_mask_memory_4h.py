"""Explicit resource-only amendment; unchanged native numerical implementation."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from scripts import probe_native_mask_memory as native
from scripts.audit_native_mask_memory_smoke import verify_raw_archive

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SMOKE = ROOT / "native-mask-memory-smoke-02"
AUDIT = ROOT / "native-mask-memory-independent-smoke-audit-01.json"
PUBLICATION = Path("docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.json")
PARENT_PINS = {
    str(SMOKE / "result.json"): "eb7a636b909107fdc3c1b6c0c0b59c3895d575a462c4a8e1ab2bd63e22541107",
    str(AUDIT): "538db188bc1195ea8bc8c5a5cd76539b57de99014b9b79b92fbd0fd1186f1263",
    str(PUBLICATION): "26ff971cf36d7ab17b875d679804768408082138faaeebbd79cfeb5fc4bc9e5b",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_4H_PROTOCOL_20261014.md")


def validate_amendment(prior, current):
    old = prior["contract"]
    if {k: v for k, v in old.items() if k != "source_commit"} != {
        k: v for k, v in current.items() if k != "source_commit"
    }:
        raise ValueError("only source-commit provenance may differ before resource extension")
    if (
        prior["status"] != "PASS_NATIVE_MEMORY_SMOKE_ONLY"
        or old["max_main_seconds"] != 7200
        or [(r["arm"], r["cache_size"]) for r in prior["rows"]]
        != [("actual", 1), ("actual", 32), ("initial_repeat", 32)]
    ):
        raise ValueError("exact original smoke resource/cohort contract required")
    projected = sum(
        r["seconds_by_seed"][-1] * 696 * 1.25 + r["seconds_by_seed"][0] * 8
        for r in prior["rows"]
        if r["cache_size"] == 32
    )
    if (
        not np.isfinite(projected)
        or not 7200 <= projected < 14400
        or projected != prior["projected_main_seconds"]
    ):
        raise ValueError("recomputed projection does not fit explicit amendment")
    return dict(
        original_scale_gate_passed=False,
        amended_scale_gate_passed=True,
        original_main_limit_seconds=7200,
        main_limit_seconds=14400,
        projected_main_seconds=projected,
        original_contract_sha256=native.digest(old),
        original_source_commit=old["source_commit"],
        numerical_conditions_changed=False,
    )


def reference_contract():
    native.verify_inherited(PARENT_PINS)
    publication = json.loads(PUBLICATION.read_bytes())
    native.verify_inherited(
        {str(Path("scripts/audit_native_mask_memory_smoke.py")): publication["auditor_sha256"]}
    )
    prior = json.loads((SMOKE / "result.json").read_bytes())
    audited = json.loads(AUDIT.read_bytes())
    if (
        audited["status"] != "PASS_INDEPENDENT_RAW_SMOKE_AUDIT_ONLY"
        or audited["scale_gate_passed"]
        or audited["predictions"] != 192
        or not prior["parameters_frozen"]
        or not prior["rng_unchanged"]
        or prior["state_before"] != prior["state_after"]
    ):
        raise ValueError("historical numerical/freeze admission incomplete")
    native.verify_inherited(audited["protected_hashes"])
    current = native.load_contract()
    amendment = validate_amendment(prior, current)
    references = {
        **PARENT_PINS,
        **audited["protected_hashes"],
        **native.evidence(SMOKE, prior, "smoke"),
    }
    banks = {}
    for row in prior["rows"]:
        with np.load(row["artifact"]["path"], allow_pickle=False) as b:
            a = {k: b[k] for k in b.files}
        verify_raw_archive(
            a, current["plans"][0]["seed_indices"][:2], current["plans"][0]["frame_indices"]
        )
        if a["raw_sha256"].tolist() != row["raw_prefix_sha256"]:
            raise ValueError("original prefix binding changed")
        banks[(row["arm"], row["cache_size"])] = a
    for k in banks[("actual", 1)]:
        if banks[("actual", 1)][k].tobytes() != banks[("actual", 32)][k].tobytes():
            raise ValueError("original native/cache parity differs")
    own = [
        Path(__file__),
        PROTOCOL,
        Path("scripts/audit_native_mask_memory_smoke.py"),
        Path("tests/test_native_mask_memory_4h.py"),
        Path("docs/superpowers/plans/2026-10-14-native-mask-memory-4h.md"),
    ]
    current = dict(
        current,
        max_main_seconds=14400,
        resource_amendment=amendment,
        source_hashes={
            **current["source_hashes"],
            **references,
            **{str(p.resolve()): native.sha(p) for p in own},
        },
    )
    prefix = {r["arm"]: r["raw_prefix_sha256"] for r in prior["rows"] if r["cache_size"] == 32}
    return current, references, prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    cap = 600 if args.replay_from else 14400
    result = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        rows=[],
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        resource_body_limit_seconds=cap,
    )
    native.write(args.output / "live-process.json", result)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            raise MemoryError("12GiB free GPU preflight required")
        native.limits(tick, args.output, cap)
        contract, references, prefix = reference_contract()
        native.write(
            args.output / "config.json",
            dict(
                contract=contract,
                mode="replay" if args.replay_from else "main",
                resource_body_limit_seconds=cap,
                command=subprocess.list2cmdline(
                    ["python", "-m", "scripts.probe_native_mask_memory_4h"] + os.sys.argv[1:]
                ),
            ),
        )
        result.update(
            contract=contract,
            contract_sha256=native.digest(contract),
            config_sha256=native.sha(args.output / "config.json"),
        )
        if args.replay_from:
            prior, replay_pins = native.admit_main(args.replay_from, contract)
            references.update(replay_pins)
        elif shutil.disk_usage(args.output).free < 8 * 2**30 + 450 * 2**20:
            raise OSError("packed-output budget plus8GiB reserve required")
        model, info = Sam2VideoModel.from_pretrained(
            native.MODEL,
            local_files_only=True,
            output_loading_info=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        if any(
            info.get(k)
            for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
        ):
            raise ValueError("strict released weight load required")
        model = model.eval().requires_grad_(False).cuda()
        if sum(p.numel() for p in model.parameters()) != 38962498:
            raise ValueError("registered parameter count changed")
        processor = Sam2VideoProcessor.from_pretrained(native.MODEL, local_files_only=True)
        before = native.state_hash(model)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all()
        numpy_rng = np.random.get_state()
        result.update(state_before=before, gpu=torch.cuda.get_device_name(), loading_info=info)
        native.write(args.output / "progress.json", result)
        for plan in contract["plans"][:1] if args.replay_from else contract["plans"]:
            native.write(args.output / "current-input-plan.json", plan)
            rgb = native.decode(plan)
            with np.load(plan["proposal_path"], allow_pickle=False) as b:
                indices = b["nms_indices"].copy()
                masks = native.seed_masks(b["masks_packed"], indices)
            if args.replay_from:
                indices, masks = indices[:2], masks[:2]
            for arm in native.ARMS:
                native.limits(tick, args.output, cap)
                values, metrics = native.execute_arm(
                    model,
                    processor,
                    rgb,
                    masks,
                    arm,
                    32,
                    tick,
                    args.output,
                    cap,
                    bool(args.replay_from),
                    prefix[arm] if plan["video_id"] == native.IDS[0] else None,
                )
                artifact = native.save_arm(
                    args.output / f"{plan['video_id']}-{arm}-cache32.npz",
                    values,
                    indices,
                    plan["frame_indices"],
                )
                row = dict(
                    video_id=plan["video_id"],
                    arm=arm,
                    cache_size=32,
                    seeds=len(indices),
                    native_object_frames=len(indices) * 32,
                    artifact=artifact,
                    **metrics,
                    raw_prefix_sha256=[v["raw_sha256"] for v in values[:2]],
                )
                result["rows"].append(row)
                result.update(seconds=time.perf_counter() - tick)
                native.write(args.output / "progress.json", result)
                if args.replay_from:
                    old = next(
                        r
                        for r in prior["rows"]
                        if r["video_id"] == plan["video_id"] and r["arm"] == arm
                    )
                    with np.load(old["artifact"]["path"], allow_pickle=False) as saved:
                        for k in ["masks_packed", "boxes", "scores", "empty_masks", "raw_sha256"]:
                            a, b = np.stack([v[k] for v in values]), saved[k][:2]
                            if (
                                a.shape != b.shape
                                or a.dtype != b.dtype
                                or a.tobytes() != b.tobytes()
                            ):
                                raise ValueError("fresh fixed prefix bytes differ from main")
                del values
            del rgb, masks
            gc.collect()
        if not args.replay_from and (
            len(result["rows"]) != 16 or sum(r["seeds"] for r in result["rows"]) != 1392
        ):
            raise ValueError("complete8video/696seed/twoarm coverage required")
        after = native.state_hash(model)
        if after != before or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen state/flags/gradients changed")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise ValueError("Torch RNG changed")
        n = np.random.get_state()
        if n[0] != numpy_rng[0] or not np.array_equal(n[1], numpy_rng[1]) or n[2:] != numpy_rng[2:]:
            raise ValueError("NumPy RNG changed")
        native.verify_inherited({**contract["source_hashes"], **references})
        if native.sha(args.output / "config.json") != result["config_sha256"]:
            raise ValueError("effective config changed")
        native.limits(tick, args.output, cap)
        result.update(
            status="PASS_FRESH_FIXED_NATIVE_MEMORY_PREFIX_ONLY"
            if args.replay_from
            else "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY",
            state_after=after,
            parameters_frozen=True,
            rng_unchanged=True,
            reference_hashes=references,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_working_set_bytes=native.process_peak_bytes(),
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - tick, ended_utc=datetime.now(UTC).isoformat())
        native.write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
