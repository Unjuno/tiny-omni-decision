"""Guarded full-input audit, matched-path cost smoke, fixed learning and exact replay."""

import argparse
import importlib.metadata
import json
import math
import os
import random
import shutil
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

from scripts.greatest_hits_initialization import ARMS
from scripts.greatest_hits_matched import digest, select_checkpoint
from scripts.greatest_hits_matched_inputs import (
    AUDIO,
    COMMON,
    PREFLIGHT,
    VIDEO,
    Pins,
    audit,
    common_banks,
    metadata,
    read,
)
from scripts.greatest_hits_matched_runtime import (
    MatchedStore,
    checkpoint,
    create_model,
    evaluate,
    metric_rows,
    probe_rows,
    update,
)
from scripts.greatest_hits_transport_store import ExplicitEventStore
from scripts.native_consumed_evaluation import common_task_gap
from scripts.native_fit_runtime import FeatureStore
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS
from scripts.native_video_cache import sha
from scripts.train_native_long_budget import write

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/pretrained_reuse/R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md"
PROBE_COUNTS = {"joint": 23707, "common": 84480}
SMOKE_STEPS = 32


def configuration(arm):
    if arm not in ARMS:
        raise ValueError("registered factorial arm required")
    return dict(
        arm=arm,
        seed=17,
        max_steps=MAX_STEPS,
        microbatches=8,
        mixture=["joint"] * 4 + ["text", "image", "audio", "video"],
        optimizer="AdamW",
        lr=0.001,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01,
        clip_norm=1.0,
        scheduler="cosine",
        warmup_updates=62,
        dtype="float32",
        tf32=False,
        deterministic=True,
        cpu_threads=4,
        evaluation_steps=list(EVALUATION_STEPS),
        performance_early_stopping=False,
        temperature=1.0,
        initialization="fitted control8192 model; fresh optimizer",
        selection="retention eligible; lowest joint task macro NLL; min_delta1e-5; earliest tie",
        probe="full planned consumed TRAIN union; fixed across all scheduled evaluations",
    )


class Selection:
    def __init__(self):
        self.initial, self.best, self.best_step, self.last = None, None, None, None

    def consider(self, step, joint_nll, retention):
        if self.initial is None:
            if step != 0:
                raise ValueError("initial retention reference at step0 required")
            self.initial = retention
        self.last = select_checkpoint(self.best, joint_nll, retention, self.initial)
        if self.last["selected"]:
            self.best, self.best_step = joint_nll, step
        return self.last

    def save(self, path, output, *, final=False):
        if self.last is None:
            raise ValueError("selection must precede checkpoint publication")
        if self.last["selected"]:
            shutil.copyfile(path, output / "best.pt")
            write(
                output / "best.json",
                dict(
                    step=self.best_step,
                    score=self.best,
                    checkpoint=path.name,
                    checkpoint_sha256=sha(path),
                ),
            )
        if final:
            shutil.copyfile(path, output / "final.pt")


def require_reference(phase, current, reference, max_seconds):
    expected = "PASS_MATCHED_AV_" + ("SMOKE" if phase == "train" else "TRAIN")
    if reference.get("status") != expected or any(
        reference.get(k) != current.get(k)
        for k in ("arm", "source_commit", "input_digest", "config")
    ):
        raise ValueError("same-source/arm/config/input passing reference required")
    if phase == "train":
        projected = reference.get("projected_full_seconds", 0)
        if not math.isfinite(projected) or projected <= 0 or max_seconds < 1.5 * projected:
            raise ValueError("recorded resource allowance must cover1.5x measured full cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "smoke", "train", "replay"], required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    if (
        args.max_seconds <= 0
        or (args.phase != "audit") != bool(args.arm)
        or (args.phase != "audit") != bool(args.audit)
        or bool(args.audit) != bool(args.audit_sha256)
        or (args.phase in ("train", "replay")) != bool(args.reference)
        or bool(args.reference) != bool(args.reference_sha256)
    ):
        raise ValueError(
            "explicit phase, arm, audit/reference hashes and resource allowance required"
        )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    if args.phase == "audit" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU audit requires CUDA hidden")
    if args.phase != "audit":
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
            raise RuntimeError("GPU occupied; do not compete")
    else:
        usage = "CPU-only audit"
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    source_pins, input_pins = Pins(), Pins()
    result = dict(
        status="RUNNING",
        phase=args.phase,
        arm=args.arm,
        pid=os.getpid(),
        command=[sys.executable, *sys.argv],
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        step=0,
        optimizer_updates=0,
        encoder_updates=0,
        gpu_preflight=usage,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        cloud_cost=0,
        max_seconds=args.max_seconds,
        resume_implemented=False,
        quality_evaluated=False,
        reference_sha256=args.reference_sha256,
        audit_sha256=args.audit_sha256,
    )
    last_check = 0.0

    def guard():
        nonlocal last_check
        elapsed = time.perf_counter() - started
        result["peak_working_set_bytes"] = psutil.Process().memory_info().peak_wset
        result["peak_vram_bytes"] = (
            torch.cuda.max_memory_allocated() if args.phase != "audit" else 0
        )
        if (
            elapsed > args.max_seconds
            or result["peak_working_set_bytes"] > 10 * 2**30
            or result["peak_vram_bytes"] > 6 * 2**30
            or shutil.disk_usage(args.output).free < 16 * 2**30
        ):
            raise RuntimeError("recorded resource guard; experiment incomplete")
        if elapsed - last_check > 15:
            source_pins.check()
            input_pins.check()
            last_check = elapsed

    def save_scores(label, scores, gates):
        with (args.output / (label + ".npy")).open("xb") as stream:
            np.save(stream, scores, allow_pickle=False)
        if gates is not None:
            with (args.output / (label + "-gates.npy")).open("xb") as stream:
                np.save(stream, gates, allow_pickle=False)

    try:
        write(args.output / "result.json", result)
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0"):
            if name:
                source_pins.add(REPO / name)
        source_pins.add(PROTOCOL)
        if args.phase == "audit":
            proof = audit(input_pins, guard)
            write(args.output / "admission.json", proof)
            result.update(
                status="PASS_MATCHED_AV_INPUT_AUDIT",
                input_digest=proof["input_digest"],
                verified_video_events=proof["verified_video_events"],
                video_smoke_matches=proof["video_smoke_matches"],
            )
        else:
            input_pins.add(args.audit / "result.json", args.audit_sha256)
            audit_receipt = read(args.audit / "result.json")
            if audit_receipt["status"] != "PASS_MATCHED_AV_INPUT_AUDIT":
                raise ValueError("complete independent input audit required")
            input_pins.add(
                args.audit / "admission.json", audit_receipt["artifacts"]["admission.json"]
            )
            proof = read(args.audit / "admission.json")
            if (
                proof["input_digest"] != digest(proof["input_pins"])
                or proof["input_digest"] != audit_receipt["input_digest"]
            ):
                raise ValueError("input admission digest differs")
            for path, expected in proof["input_pins"].items():
                input_pins.add(path, expected)
                guard()
            config = configuration(args.arm)
            config.update(
                plan_sha256=sha(PREFLIGHT / "sampling.json"), protocol_sha256=sha(PROTOCOL)
            )
            result.update(config=config, input_digest=proof["input_digest"])
            write(args.output / "effective-config.json", config)
            config_sha = sha(args.output / "effective-config.json")
            reference = None
            if args.reference:
                input_pins.add(args.reference / "result.json", args.reference_sha256)
                reference = read(args.reference / "result.json")
                require_reference(args.phase, result, reference, args.max_seconds)
                for name, expected in reference["artifacts"].items():
                    if name != Path(name).name:
                        raise ValueError("local reference artifact basename required")
                    input_pins.add(args.reference / name, expected)
            data = metadata()
            plan = read(PREFLIGHT / "sampling.json")
            if len(plan["order"]) != MAX_STEPS or digest(plan["order"]) != plan["order_sha256"]:
                raise ValueError("complete frozen sampling plan required")
            probes = probe_rows(plan["order"], data["joint", "train"], data["common", "train"])
            if {k: len(v) for k, v in probes.items()} != PROBE_COUNTS:
                raise ValueError("all108187 unique planned TRAIN questions required")
            all_rows = {r["id"]: r for k in ("joint", "common") for r in data[k, "train"]}
            coverage = read(PREFLIGHT / "coverage.json")
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if not torch.cuda.is_available():
                raise RuntimeError("existing local CUDA required")
            torch.cuda.reset_peak_memory_stats()
            model = create_model(args.arm)
            model.load_state_dict(
                torch.load(
                    PREFLIGHT / (args.arm + "-initial.pt"), weights_only=True, map_location="cpu"
                ),
                strict=True,
            )
            count = sum(p.numel() for p in model.parameters())
            if count != (889345 if args.arm == "C0" else 1732420) or 195365529 + count > 200000000:
                raise ValueError("encoder-inclusive parameter ledger differs")
            model.cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
            )
            store = MatchedStore(
                ExplicitEventStore(
                    proof["binding"],
                    dict(audio=AUDIO, video=VIDEO),
                    dict(
                        audio=read(AUDIO / "result.json")["artifacts"],
                        video=read(VIDEO / "result.json")["artifacts"],
                    ),
                    device="cuda",
                ),
                FeatureStore(
                    read(COMMON / "observations.json"),
                    common_banks(),
                    read(COMMON / "query-map.json"),
                ),
            )
            result.update(
                versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "psutil")},
                python=sys.version,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(),
                trainable_parameters=count,
                encoder_inclusive_parameters=195365529 + count,
                config_sha256=config_sha,
                setup_seconds=time.perf_counter() - started,
                attention_backend_policy=dict(
                    flash=torch.backends.cuda.flash_sdp_enabled(),
                    memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    math=torch.backends.cuda.math_sdp_enabled(),
                    cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                    selection="PyTorch automatic",
                ),
            )
            if reference and (
                reference["versions"] != result["versions"]
                or reference["attention_backend_policy"] != result["attention_backend_policy"]
            ):
                raise ValueError("matching runtime/backend required")
            evaluation_sets = {
                "train-joint": (probes["joint"], "joint"),
                "train-common": (probes["common"], "joint"),
                "validation-joint": (data["joint", "validation"], "joint"),
                "validation-audio": (data["joint", "validation"], "audio"),
                "validation-video": (data["joint", "validation"], "video"),
                "validation-common": (data["common", "validation"], "joint"),
            }
            write(
                args.output / "evaluation-rows.json",
                {
                    label: [
                        {
                            k: r[k]
                            for k in (
                                "id",
                                "split",
                                "options",
                                "target_index",
                                "source",
                                "task",
                                "modality",
                                "event_id",
                                "parent",
                                "observation",
                                "group",
                            )
                            if k in r
                        }
                        for r in metric_rows(rows)
                    ]
                    for label, (rows, _) in evaluation_sets.items()
                },
            )
            selector, curve = Selection(), []
            training_seconds, evaluation_seconds = 0.0, 0.0

            def full_evaluation(step, *, replay_from=None, prefix=None):
                nonlocal evaluation_seconds
                tick = time.perf_counter()
                summaries, matrices = {}, {}
                for label, (rows, mode) in evaluation_sets.items():
                    part = time.perf_counter()
                    scores, gates, summary = evaluate(model, rows, store, mode=mode, guard=guard)
                    name = f"step-{step:05}-{label}"
                    if replay_from:
                        np.testing.assert_array_equal(
                            scores, np.load(replay_from / (name + ".npy"), allow_pickle=False)
                        )
                        if gates is not None:
                            np.testing.assert_array_equal(
                                gates,
                                np.load(replay_from / (name + "-gates.npy"), allow_pickle=False),
                            )
                    save_scores((prefix + "-" if prefix else "") + name, scores, gates)
                    summaries[label], matrices[label] = summary, scores
                    print(
                        "evaluation",
                        label,
                        step,
                        "rows",
                        len(rows),
                        "seconds",
                        round(time.perf_counter() - part, 2),
                        flush=True,
                    )
                gaps = {
                    kind: common_task_gap(
                        metric_rows(probes[kind]),
                        matrices["train-" + kind],
                        metric_rows(data[kind, "validation"]),
                        matrices["validation-" + kind],
                    )
                    for kind in ("joint", "common")
                }
                seconds = time.perf_counter() - tick
                evaluation_seconds += seconds
                return dict(step=step, metrics=summaries, gaps=gaps, seconds=seconds)

            def save_checkpoint(step):
                path = args.output / f"checkpoint-{step:05}.pt"
                if path.exists():
                    raise ValueError("immutable checkpoint already exists")
                value = checkpoint(
                    model,
                    optimizer,
                    step=step,
                    best_step=selector.best_step,
                    best_score=selector.best,
                    plan_sha256=config["plan_sha256"],
                    config_sha256=config_sha,
                )
                value["selector_initial"] = selector.initial
                temporary = path.with_suffix(".pt.tmp")
                torch.save(value, temporary)
                temporary.replace(path)
                return path

            def scheduled(step):
                item = full_evaluation(step)
                common = item["metrics"]["validation-common"]
                joint = item["metrics"]["validation-joint"]
                item["selection"] = selector.consider(step, joint["macro"]["nll"], common)
                item.update(
                    best_step=selector.best_step, accounting=coverage[str(step)], warnings=[]
                )
                if curve:
                    previous = curve[-1]["metrics"]
                    for kind in ("joint", "common"):
                        before = previous["validation-" + kind]["macro"]
                        after = item["metrics"]["validation-" + kind]["macro"]
                        if (
                            item["metrics"]["train-" + kind]["macro"]["nll"]
                            < previous["train-" + kind]["macro"]["nll"]
                            and after["nll"] > before["nll"]
                        ):
                            item["warnings"].append(
                                kind + ": TRAIN NLL fell while validation NLL rose"
                            )
                        if after["accuracy"] > before["accuracy"] and after["ece"] > before["ece"]:
                            item["warnings"].append(kind + ": Accuracy rose with worse ECE")
                item["warnings"] += item["selection"]["reasons"]
                curve.append(item)
                path = save_checkpoint(step)
                selector.save(path, args.output, final=step == MAX_STEPS)
                write(args.output / "learning-curve.json", curve)
                result.update(
                    best_step=selector.best_step,
                    best_score=selector.best,
                    selected_checkpoint_accounting=coverage[str(selector.best_step)],
                    training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds,
                )
                write(args.output / "result.json", result)
                print(
                    "scheduled",
                    step,
                    "joint",
                    joint["macro"],
                    "best",
                    selector.best_step,
                    flush=True,
                )

            if args.phase == "replay":
                for name in ("best", "final"):
                    saved = torch.load(
                        args.reference / (name + ".pt"), map_location="cpu", weights_only=True
                    )
                    if (
                        saved["sampler_sha256"] != config["plan_sha256"]
                        or saved["config_sha256"] != config_sha
                        or saved["step"]
                        != (reference["best_step"] if name == "best" else MAX_STEPS)
                    ):
                        raise ValueError("selected/final checkpoint binding differs")
                    model.load_state_dict(saved["model"], strict=True)
                    item = full_evaluation(saved["step"], replay_from=args.reference, prefix=name)
                    write(args.output / (name + "-reload.json"), item)
                result["best_final_reload_verified"] = True
            else:
                if args.phase == "train":
                    scheduled(0)
                timings = []
                updates = SMOKE_STEPS if args.phase == "smoke" else MAX_STEPS
                with (args.output / "training-history.jsonl").open(
                    "x", encoding="utf-8"
                ) as history:
                    for step in range(1, updates + 1):
                        guard()
                        tick = time.perf_counter()
                        item = update(
                            model,
                            [all_rows[i] for i in plan["order"][step - 1]],
                            store,
                            step,
                            optimizer,
                        )
                        torch.cuda.synchronize()
                        duration = time.perf_counter() - tick
                        timings.append(duration)
                        training_seconds += duration
                        item.update(
                            seconds=duration,
                            elapsed_seconds=time.perf_counter() - started,
                            peak_vram_bytes=torch.cuda.max_memory_allocated(),
                            completed_utc=datetime.now(UTC).isoformat(),
                        )
                        history.write(json.dumps(item, allow_nan=False) + "\n")
                        history.flush()
                        result.update(step=step, optimizer_updates=step)
                        if args.phase == "train" and step in EVALUATION_STEPS:
                            scheduled(step)
                        if step % 32 == 0:
                            write(args.output / "result.json", result)
                            print("update", step, "/", updates, "CE", item["mean_ce"], flush=True)
                if args.phase == "smoke":
                    path = save_checkpoint(SMOKE_STEPS)
                    shutil.copyfile(path, args.output / "final.pt")
                    item = full_evaluation(SMOKE_STEPS)
                    write(args.output / "smoke-evaluation.json", item)
                    # Full path includes two endpoint reloads in addition to ten curve points.
                    mean_update = float(np.mean(timings[min(4, len(timings) - 1) :]))
                    projected = MAX_STEPS * mean_update + 12 * item["seconds"]
                    projected += result["setup_seconds"]
                    result.update(
                        projected_full_seconds=projected,
                        projected_training_seconds=MAX_STEPS * mean_update,
                        projected_evaluation_and_reload_seconds=12 * item["seconds"],
                        measured_full_evaluation_seconds=item["seconds"],
                        update_seconds=timings,
                    )
                    model.load_state_dict(
                        torch.load(path, map_location="cpu", weights_only=True)["model"],
                        strict=True,
                    )
                    write(
                        args.output / "smoke-reload.json",
                        full_evaluation(SMOKE_STEPS, replay_from=args.output, prefix="reload"),
                    )
                    result["smoke_reload_verified"] = True
                else:
                    for name, step in (("best", selector.best_step), ("final", MAX_STEPS)):
                        saved = torch.load(
                            args.output / (name + ".pt"), map_location="cpu", weights_only=True
                        )
                        model.load_state_dict(saved["model"], strict=True)
                        write(
                            args.output / (name + "-reload.json"),
                            full_evaluation(step, replay_from=args.output, prefix=name),
                        )
                    result.update(
                        best_final_reload_verified=True, full_accounting=coverage[str(MAX_STEPS)]
                    )
                result.update(
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds
                )
            store.joint.verify_files()
            result.update(status="PASS_MATCHED_AV_" + args.phase.upper(), quality_evaluated=True)
        input_pins.verify()
        source_pins.verify()
        guard()
    except BaseException as exc:
        result.update(status="INCOMPLETE", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started,
            finished_utc=datetime.now(UTC).isoformat(),
            source_pins=source_pins.hashes,
            artifacts={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        )
        write(args.output / "result.json", result)
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
