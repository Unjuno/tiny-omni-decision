"""Pinned common native head fit, TRAIN-only smoke, or independent checkpoint replay."""

import argparse
import hashlib
import importlib.metadata
import json
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

from scripts.native_fit import (
    accounting,
    choose_best,
    learning_rate,
    option_order,
    sample_order,
    summarize,
)
from scripts.native_fit_runtime import FeatureStore
from scripts.native_video_cache import sha
from tiny_omni_decision.native_candidate import NativeCandidateReadout
from tiny_omni_decision.native_temporal import TemporalNativeCandidate

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_common_fit_v1.json"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify(pins):
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError("changed input/source: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "train", "replay"], required=True)
    parser.add_argument("--variant", choices=["baseline", "lstm"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase != "smoke") != bool(args.reference) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("train needs smoke receipt; replay needs train receipt and SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, utilization = [int(v.strip()) for v in usage.split(",")]
    if memory > 512 or utilization > 5:
        raise RuntimeError("GPU occupied; do not compete")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(
        status="RUNNING",
        phase=args.phase,
        variant=args.variant,
        pid=os.getpid(),
        command=sys.argv,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        config_sha256=sha(CONFIG),
        started_utc=datetime.now(UTC).isoformat(),
        step=0,
        best_step=None,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        cloud_cost=0,
        validation_loaded=args.phase != "smoke",
        encoder_updates=0,
        gpu_preflight=usage,
    )
    write(args.output / "result.json", result)
    try:
        config = json.loads(CONFIG.read_text())
        if config["training"] != {
            "seed": 17,
            "max_steps": 2048,
            "microbatches_per_update": 4,
            "lr": 0.001,
            "warmup_updates": 62,
            "scheduler": "cosine",
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "clip_norm": 1.0,
            "evaluation_interval": 256,
        }:
            raise ValueError("configuration differs from the implemented fixed schedule")
        pins = dict(config["input_pins"])
        pins[str(CONFIG)] = sha(CONFIG)
        source_names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in source_names if name})
        verify(pins)
        corpus = Path(config["corpus"])
        queries = Path(config["queries"])
        receipt = json.loads((corpus / "result.json").read_text())
        query_receipt = json.loads((queries / "result.json").read_text())
        if (
            receipt["status"] != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
            or query_receipt["status"] != "PASS_NATIVE_COMMON_QUERIES_CACHE"
        ):
            raise ValueError("complete source corpus/query cache required")
        pins.update(receipt["validated_hashes"])
        pins.update({str(corpus / name): digest for name, digest in receipt["artifacts"].items()})
        pins.update(
            {str(queries / name): digest for name, digest in query_receipt["artifacts"].items()}
        )
        verify(pins)
        reference = None
        if args.reference:
            if sha(args.reference / "result.json") != args.reference_sha256:
                raise ValueError("reference receipt hash differs")
            reference = json.loads((args.reference / "result.json").read_text())
            expected = "PASS_NATIVE_FIT_" + ("SMOKE" if args.phase == "train" else "TRAIN")
            if (
                reference["status"] != expected
                or reference["validated_hashes"] != pins
                or reference["config_sha256"] != sha(CONFIG)
                or reference["variant"] != args.variant
            ):
                raise ValueError("matching completed reference required")
            verify(
                {
                    str(args.reference / name): digest
                    for name, digest in reference["artifacts"].items()
                }
            )
            if args.phase == "train" and reference["projected_full_seconds"] > 7200:
                raise RuntimeError("registered local runtime budget is not practical")
        train = [
            json.loads(line)
            for line in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        validation = (
            []
            if args.phase == "smoke"
            else [
                json.loads(line)
                for line in (corpus / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        )
        if (
            len(train) != 100580
            or (validation and len(validation) != 24346)
            or any(r["split"] != "validation" for r in validation)
        ):
            raise ValueError("full registered development corpus required")
        observations = json.loads((corpus / "observations.json").read_text())
        banks = json.loads((corpus / "banks.json").read_text())
        banks["new/queries.npy"] = dict(
            path=str(queries / "queries.npy"),
            sha256=query_receipt["artifacts"]["queries.npy"],
            shape=[2622, 384],
        )
        mapping = json.loads((corpus / "query-map.json").read_text())
        order = sample_order(train, 2048)
        full_sampling = accounting([train[i] for batch in order for i in batch])
        write(
            args.output / "sampling.json",
            dict(
                order=order,
                accounting=full_sampling,
                option_permutation="seed17:step:original-row-id SHA256-seeded Python shuffle",
            ),
        )
        if (
            reference
            and sha(args.output / "sampling.json") != reference["artifacts"]["sampling.json"]
        ):
            raise ValueError("reference sampler order/accounting differs")
        probe_indices = []
        groups = {}
        for i, row in enumerate(train):
            groups.setdefault((row["source"], row["task"]), []).append(i)
        for key in sorted(groups):
            probe_indices.extend(
                sorted(
                    groups[key],
                    key=lambda i: hashlib.sha256(
                        f"17:train-probe:{train[i]['id']}".encode()
                    ).hexdigest(),
                )[:16]
            )
        probe = [train[i] for i in probe_indices]
        write(args.output / "train-probe-ids.json", [r["id"] for r in probe])
        if validation:
            write(
                args.output / "validation-ids.json",
                [
                    dict(id=r["id"], option_ids=r["option_ids"], target_index=r["target_index"])
                    for r in validation
                ],
            )
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
        torch.cuda.reset_peak_memory_stats()
        base = NativeCandidateReadout()
        base.load_state_dict(
            torch.load(config["initial_readout"], map_location="cpu", weights_only=True)
        )
        model = (base if args.variant == "baseline" else TemporalNativeCandidate(base)).cuda()
        parameters = sum(p.numel() for p in model.parameters())
        if 195365529 + parameters > 200000000:
            raise ValueError("prequantization cap exceeded")
        store = FeatureStore(observations, banks, mapping)
        result.update(
            validated_hashes=pins,
            trainable_parameters=parameters,
            total_parameters=195365529 + parameters,
            full_planned_accounting=full_sampling,
            train_probe_count=len(probe),
            validation_count=len(validation),
            versions={p: importlib.metadata.version(p) for p in ["torch", "numpy", "psutil"]},
            python=sys.version,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            dtype="float32",
            tf32=False,
            initialization="original native seed17 readout",
            data_setup_seconds=time.perf_counter() - started,
        )
        if reference and reference["versions"] != result["versions"]:
            raise ValueError("reference runtime changed")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        best = None
        best_step = None
        history = []
        evaluation_seconds = 0.0
        training_seconds = 0.0

        def guard():
            mem = psutil.Process().memory_info()
            result["peak_working_set_bytes"] = mem.peak_wset
            if (
                time.perf_counter() - started > 7200
                or mem.peak_wset > 10 * 2**30
                or torch.cuda.max_memory_allocated() > 6 * 2**30
                or shutil.disk_usage(args.output).free < 2 * 2**30
            ):
                raise RuntimeError("registered resource budget exceeded")

        def checkpoint(step):
            rng = np.random.get_state()
            value = dict(
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                step=step,
                best_step=best_step,
                best_score=best,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),
                numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]),
                sampler_file_sha256=sha(args.output / "sampling.json"),
                config_sha256=sha(CONFIG),
            )
            path = args.output / f"checkpoint-{step:04}.pt"
            torch.save(value, path)
            return path

        def evaluate(rows, label):
            nonlocal evaluation_seconds
            tick = time.perf_counter()
            model.eval()
            scores = np.full((len(rows), 40), -1e30, dtype=np.float32)
            with torch.no_grad():
                for i, row in enumerate(rows):
                    blocks, q = store.get(row)
                    values = model(blocks, q).cpu().numpy()
                    scores[i, : len(values)] = values
                    if (i + 1) % 2048 == 0:
                        guard()
                        print("evaluate", label, i + 1, "/", len(rows), flush=True)
            torch.cuda.synchronize()
            evaluation_seconds += time.perf_counter() - tick
            return scores, summarize(rows, scores)

        def scheduled_evaluation(step):
            nonlocal best, best_step
            probe_scores, train_metrics = evaluate(probe, f"{step}-train")
            scores, metrics = evaluate(validation, f"{step}-validation")
            np.save(args.output / f"train-probe-{step:04}.npy", probe_scores, allow_pickle=False)
            np.save(args.output / f"validation-{step:04}.npy", scores, allow_pickle=False)
            selected = choose_best(best, metrics["macro"]["nll"])
            if selected:
                best = metrics["macro"]["nll"]
                best_step = step
            warnings = []
            if history:
                old = history[-1]
                if (
                    train_metrics["macro"]["nll"] < old["train_probe"]["macro"]["nll"]
                    and metrics["macro"]["nll"] > old["validation"]["macro"]["nll"]
                ):
                    warnings.append("train probe NLL falling while validation macro NLL rises")
                if (
                    metrics["macro"]["accuracy"] > old["validation"]["macro"]["accuracy"]
                    and metrics["minimum_modality_accuracy"]
                    < old["validation"]["minimum_modality_accuracy"]
                ):
                    warnings.append("aggregate gain with weakest-modality degradation")
            used = [train[i] for batch in order[:step] for i in batch]
            item = dict(
                step=step,
                train_probe=train_metrics,
                validation=metrics,
                selected=selected,
                best_step=best_step,
                warnings=warnings,
                accounting=accounting(used),
            )
            history.append(item)
            write(args.output / "learning-curve.json", history)
            path = checkpoint(step)
            if selected:
                shutil.copyfile(path, args.output / "best.pt")
                write(args.output / "best.json", dict(step=step, score=best, checkpoint=path.name))
            result.update(
                step=step,
                best_step=best_step,
                best_score=best,
                evaluation_seconds=evaluation_seconds,
                training_seconds=training_seconds,
            )
            write(args.output / "result.json", result)
            print("validation", step, "macro", metrics["macro"], "best", best_step, flush=True)

        if args.phase == "replay":
            for label, step in (("best", reference["best_step"]), ("final", 2048)):
                checkpoint_data = torch.load(
                    args.reference / (label + ".pt"), map_location="cpu", weights_only=True
                )
                if checkpoint_data["step"] != step:
                    raise ValueError("checkpoint step differs")
                model.load_state_dict(checkpoint_data["model"], strict=True)
                for split, rows in (("validation", validation), ("train-probe", probe)):
                    scores, metrics = evaluate(rows, label + "-" + split)
                    expected = np.load(
                        args.reference / f"{split}-{step:04}.npy", allow_pickle=False
                    )
                    np.testing.assert_array_equal(scores, expected)
                    write(args.output / f"{label}-{split}-metrics.json", metrics)
            result.update(
                step=2048,
                best_step=reference["best_step"],
                replay_max_abs=0.0,
                reference_sha256=args.reference_sha256,
            )
        else:
            torch.save(model.state_dict(), args.output / "initial.pt")
            if args.phase == "train":
                scheduled_evaluation(0)
            limit = 32 if args.phase == "smoke" else 2048
            steps = []
            with (args.output / "training-history.jsonl").open("x", encoding="utf-8") as log:
                for step in range(1, limit + 1):
                    guard()
                    tick = time.perf_counter()
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    lr = learning_rate(step)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    losses = []
                    correct = 0
                    for index in order[step - 1]:
                        row = train[index]
                        blocks, q = store.get(row)
                        perm = option_order(row, step)
                        target = perm.index(row["target_index"])
                        scores = model(blocks, q[perm])
                        loss = -scores.log_softmax(0)[target]
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        (loss / 4).backward()
                        losses.append(float(loss.detach()))
                        correct += int(scores.detach().argmax().item() == target)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise ValueError("nonfinite gradient norm")
                    optimizer.step()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - tick
                    training_seconds += seconds
                    steps.append(seconds)
                    item = dict(
                        step=step,
                        ce=float(np.mean(losses)),
                        accuracy=correct / 4,
                        lr=lr,
                        gradient_norm_before_clip=float(norm),
                        seconds=seconds,
                        sample_ids=[train[i]["id"] for i in order[step - 1]],
                    )
                    log.write(json.dumps(item, allow_nan=False) + "\n")
                    if step % 32 == 0:
                        log.flush()
                        result.update(step=step, training_seconds=training_seconds)
                        write(args.output / "result.json", result)
                        print("step", step, "ce", item["ce"], "sec", seconds, flush=True)
                    if args.phase == "train" and step % 256 == 0:
                        scheduled_evaluation(step)
            if args.phase == "smoke":
                scores, metrics = evaluate(probe, "smoke-train")
                torch.save(model.state_dict(), args.output / "smoke-final.pt")
                model.load_state_dict(torch.load(args.output / "smoke-final.pt", weights_only=True))
                repeat, _ = evaluate(probe, "smoke-reload")
                np.testing.assert_array_equal(scores, repeat)
                np.save(args.output / "smoke-probe.npy", scores, allow_pickle=False)
                result.update(
                    smoke_train_probe=metrics,
                    smoke_reload_max_abs=0,
                    median_seconds_per_update=float(np.median(steps[4:])),
                    projected_full_seconds=float(
                        np.median(steps[4:]) * 2048
                        + (evaluation_seconds / 2) * 9 * (24346 + len(probe)) / len(probe) * 2
                        + 60
                    ),
                )
            else:
                shutil.copyfile(args.output / "checkpoint-2048.pt", args.output / "final.pt")
                result.update(
                    best_step=best_step,
                    best_score=best,
                    selection="hierarchical macro NLL",
                    final_metrics=history[-1]["validation"],
                    selected_metrics=next(
                        h["validation"] for h in history if h["step"] == best_step
                    ),
                    selected_accounting=accounting(
                        [train[i] for b in order[:best_step] for i in b]
                    ),
                )
            result.update(
                step=limit,
                actual_accounting=accounting([train[i] for b in order[:limit] for i in b]),
            )
        guard()
        verify(pins)
        result.update(
            status="PASS_NATIVE_FIT_" + args.phase.upper(),
            training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.name != "result.json"},
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", failure=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        write(args.output / "result.json", result)
    print(result["status"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
