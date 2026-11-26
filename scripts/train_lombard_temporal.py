"""Fixed cache-only CPU temporal comparison; no model/encoder/source overrides."""

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.clap_progress_io import save_progress
from scripts.fetch_robomnist_input_smoke import peak_bytes
from scripts.lombard_readout_core import (
    SIZES,
    fit_normalizer,
    json_decision,
    select_temperatures,
    selected_epoch,
    summarize,
)
from scripts.lombard_readout_data import load_data, read_json, sha
from scripts.lombard_temporal_core import visual_condition
from scripts.lombard_temporal_model import TemporalSlotHead
from scripts.robomnist_audio_cue import recheck

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
CONFIG = Path("configs/pretrained_reuse/lombard-grid-temporal-v1.json")
VIDEO_SHA = "3cbf602dcb9e659f65e4c1562724072c80f6d94f0093d2e543ab61cd2e8d7f07"
FRAME_SHA = "a4f44cd94ef762316440e359f5e916a0e862b2e61378aeafa0bd6f4e9f67699a"
ARMS = ["audio", "video_broadcast", "video_ordered", "joint_broadcast", "joint_ordered"]


def orders_for(count, epochs):
    if count < 1 or epochs < 1:
        raise ValueError("positive sampler dimensions required")
    g = torch.Generator().manual_seed(17)
    return torch.stack([torch.randperm(count, generator=g) for _ in range(epochs)])


def guard(output, tick, reserve=0):
    elapsed, peak = time.perf_counter() - tick, peak_bytes()
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if (
        elapsed > 7200
        or peak > 3 * 2**30
        or size + reserve + 2**22 > 128 * 2**20
        or shutil.disk_usage(output).free < 2**30
    ):
        raise RuntimeError("fixed temporal CPU/time/output/disk resource guard exhausted")
    return dict(seconds=elapsed, peak_cpu_bytes=peak, output_bytes=size)


def state_sha(model):
    import hashlib

    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_scope(smoke):
    raw, targets, rows, provenance = load_data(ROOT, smoke=smoke)
    folder = ROOT / "lombard-grid-videoprism-native-features-01"
    info = read_json(folder / "result.json", VIDEO_SHA)
    if info["status"] != "PASS" or info["feature_hashes"]["framewise.npy"] != FRAME_SHA:
        raise ValueError("full native visual authority required")
    p = folder / "framewise.npy"
    if sha(p) != FRAME_SHA:
        raise ValueError("native chronological visual bank hash differs")
    frames = np.load(p, mmap_mode="r", allow_pickle=False)
    if frames.shape != (5367, 16, 768) or frames.dtype != np.float32:
        raise ValueError("complete native chronological bank shape required")
    ids = [r["source_id"] for r in info["rows"]]
    if len(ids) != 5367 or len(set(ids)) != 5367:
        raise ValueError("unique full frame-bank IDs required")
    if smoke:
        position = {source: i for i, source in enumerate(ids)}
        selected = frames[[position[r["source_id"]] for r in rows["train"]]]
        framed = {"train": selected}
    else:
        if ids != [r["source_id"] for s in rows for r in rows[s]]:
            raise ValueError("full chronological frame/row ordering mismatch")
        framed, begin = {}, 0
        for split in rows:
            framed[split] = frames[begin : begin + len(rows[split])]
            begin += len(rows[split])
    for split, value in framed.items():
        visual_condition(value, raw[split][1], "ordered")
    provenance.update(
        framewise_sha256=FRAME_SHA,
        frame_report_sha256=VIDEO_SHA,
        frames=16,
        temporal_mean_exact=True,
    )
    return raw, framed, targets, rows, provenance


def bound_sources():
    paths = [
        Path(__file__),
        CONFIG,
        Path("scripts/lombard_temporal_core.py"),
        Path("scripts/lombard_temporal_model.py"),
        Path("tests/test_lombard_temporal.py"),
        Path("scripts/lombard_readout_data.py"),
        Path("scripts/lombard_readout_core.py"),
        Path("scripts/lombard_grid_corpus.py"),
        Path("scripts/robomnist_audio_cue.py"),
        Path("scripts/fetch_robomnist_input_smoke.py"),
        Path("scripts/diagnose_robomnist_lengths.py"),
        Path("scripts/clap_progress_io.py"),
        Path("docs/pretrained_reuse/R2_LOMBARD_TEMPORAL_PROTOCOL_20261015.md"),
        Path("docs/superpowers/plans/2026-10-15-lombard-temporal-readout.md"),
    ]
    # Bind installed implementations, including lazily imported optimizer code.
    import importlib.util

    package_roots = [Path(torch.__file__).parent, Path(np.__file__).parent]
    spec = importlib.util.find_spec("sympy")
    if spec is not None and spec.origin:
        package_roots.append(Path(spec.origin).parent)
    for folder in package_roots:
        paths.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".pyd"})
    for folder in [
        Path(torch.__file__).parent / "lib",
        Path(np.__file__).parent.parent / "numpy.libs",
    ]:
        paths.extend(p for p in folder.glob("*") if p.suffix in {".dll", ".pyd"})
    return {str(p.resolve()): sha(p) for p in paths}


def tensors(raw, framed, normalizer, arm):
    modality = arm.split("_")[0]
    order = "ordered" if arm.endswith("ordered") else "broadcast"
    value = {}
    for split, (audio, pooled) in raw.items():
        frames = visual_condition(framed[split], pooled, order)
        a = (audio - normalizer["audio_mean"]) / normalizer["audio_scale"]
        v = (frames - normalizer["video_mean"]) / normalizer["video_scale"]
        mask = np.tile(
            np.array([modality != "video", modality != "audio"], np.float32), (len(a), 1)
        )
        if modality == "video":
            a = np.zeros_like(a)
        if modality == "audio":
            v = np.zeros_like(v)
        value[split] = tuple(torch.from_numpy(np.ascontiguousarray(z)) for z in (a, v, mask))
    return value


def predict(model, values):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(*(v[i : i + 128] for v in values)).numpy()
                for i in range(0, len(values[0]), 128)
            ]
        )


def store_prediction(output, name, logits, targets, rows, temperatures=None):
    path = output / (name + ".npz")
    if path.exists():
        raise ValueError("new prediction snapshot required")
    metrics, probabilities = summarize(logits, targets, temperatures)
    np.savez_compressed(
        path,
        logits=logits,
        probabilities=probabilities,
        targets=targets,
        source_ids=np.array([r["source_id"] for r in rows]),
        speaker_groups=np.array([r["speaker_group"] for r in rows]),
    )
    with np.load(path, allow_pickle=False) as saved:
        if (
            not np.array_equal(saved["logits"], logits)
            or not np.array_equal(saved["probabilities"], probabilities)
            or not np.array_equal(saved["targets"], targets)
        ):
            raise ValueError("prediction canonical readback differs")
    return metrics


def save_state(model, optimizer, output, name, updates, epoch):
    torch.save(model.state_dict(), output / (name + ".pt"))
    torch.save(
        dict(
            optimizer=optimizer.state_dict(),
            global_updates=updates,
            epoch=epoch,
            torch_rng=torch.get_rng_state(),
        ),
        output / (name + "-optimizer.pt"),
    )


def timing_probe(raw, framed, normalizer):
    values = tensors(raw, framed, normalizer, "joint_ordered")["train"]
    indices = torch.arange(128) % len(values[0])
    torch.manual_seed(17)
    model = TemporalSlotHead()
    before, seconds = state_sha(model), []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        result = model(*(v[indices] for v in values))
        result.square().mean().backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite timing-probe gradient")
        seconds.append(time.perf_counter() - tick)
    if state_sha(model) != before:
        raise ValueError("gradient-only timing probe changed weights")
    return dict(
        batch=128,
        replicates=3,
        seconds=seconds,
        consumed_presentations=384,
        unique_train_assets=len(values[0]),
        optimizer_updates=0,
        weights_unchanged=True,
        projected_full_seconds=768 * 5 * max(seconds) * 2 + 1000,
    )


def admit_smoke(reference, pins):
    if (
        reference.get("status") != "PASS_FIVE_ARM_TRAIN_ONLY_SMOKE"
        or reference.get("smoke") is not True
        or reference.get("gpu_used") is not False
        or reference.get("validated_hashes") != pins
        or reference.get("timing_probe", {}).get("projected_full_seconds", math.inf) >= 7200
        or any(
            r.get("validation_evaluated") or r.get("calibration_evaluated")
            for r in reference.get("arms", {}).values()
        )
        or list(reference.get("arms", {})) != ARMS
    ):
        raise ValueError("complete unchanged-code train-only smoke and runtime admission required")


def fit_arm(raw, framed, targets, rows, config, arm, output, run, tick, smoke):
    output.mkdir(exist_ok=False)
    normalizer = fit_normalizer(*raw["train"])
    np.savez(output / "normalizer.npz", **normalizer)
    values = tensors(raw, framed, normalizer, arm)
    y = torch.from_numpy(targets["train"])
    torch.manual_seed(17)
    model = TemporalSlotHead()
    if sum(p.numel() for p in model.parameters()) != 106355:
        raise ValueError("fixed readout parameter count differs")
    initial = state_sha(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    epochs = 2 if smoke else 24
    orders = orders_for(len(rows["train"]), epochs)
    np.save(output / "sample-orders.npy", orders.numpy())
    report = dict(
        status="RUNNING",
        arm=arm,
        initial_tensor_sha256=initial,
        sampler_sha256=sha(output / "sample-orders.npy"),
        history=[],
        head_parameters=106355,
        normalization_coefficients=2304,
        source_encoder_parameters={"audio": 8208384, "video": 114365184, "joint": 122573568}[
            arm.split("_")[0]
        ],
        train_unique_assets=len(rows["train"]),
        updates=0,
        validation_evaluated=False,
        calibration_evaluated=False,
        train_speaker_groups=len({r["speaker_group"] for r in rows["train"]}),
        train_condition_counts={
            c: sum(r["condition"] == c for r in rows["train"]) for c in ["p", "l"]
        },
        optimizer_reset_fresh_initialization=True,
        strict_resume_supported=False,
    )
    best = math.inf
    steps = (output / "optimizer-history.jsonl").open("x", encoding="utf-8")
    try:
        for epoch, order in enumerate(orders, 1):
            epoch_tick = time.perf_counter()
            model.train()
            total_loss, consumed, norms = 0.0, 0, []
            for indices in order.split(128):
                guard(run, tick)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*(v[indices] for v in values["train"]))
                loss = torch.stack(
                    [
                        F.cross_entropy(z, y[indices, j])
                        for j, z in enumerate(logits.split(SIZES, -1))
                    ]
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()])
                )
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError("nonfinite updated weight")
                report["updates"] += 1
                consumed += len(indices)
                total_loss += float(loss.detach()) * len(indices)
                norms.append(float(norm))
                steps.write(
                    json.dumps(
                        dict(
                            update=report["updates"],
                            epoch=epoch,
                            samples=len(indices),
                            learning_rate=0.001,
                            gradient_norm=float(norm),
                            seconds=time.perf_counter() - tick,
                        )
                    )
                    + "\n"
                )
            if consumed != len(rows["train"]):
                raise ValueError("sampler consumption differs")
            entry = dict(
                epoch=epoch,
                rolling_ce=total_loss / consumed,
                consumed_assets=epoch * consumed,
                gradient_norm_mean=float(np.mean(norms)),
                gradient_norm_max=max(norms),
                seconds=time.perf_counter() - epoch_tick,
            )
            if not smoke:
                train_logits = predict(model, values["train"])
                entry["train_fixed"] = summarize(train_logits, targets["train"])[0]
                val_logits = predict(model, values["validation"])
                entry["validation"] = store_prediction(
                    output,
                    f"epoch-{epoch:02}",
                    val_logits,
                    targets["validation"],
                    rows["validation"],
                )
                report["validation_evaluated"] = True
                score = entry["validation"]["macro"]["nll"]
                if score < best:
                    best = score
                    save_state(model, optimizer, output, "best", report["updates"], epoch)
                print(arm, "epoch", epoch, "NLL", score, flush=True)
            report["history"].append(entry)
            steps.flush()
            save_progress(output / "progress.json", report)
        save_state(model, optimizer, output, "final", report["updates"], epochs)
        if report["updates"] != epochs * math.ceil(len(rows["train"]) / 128):
            raise ValueError("optimizer update accounting differs")
        report.update(
            consumed_assets=epochs * len(rows["train"]),
            repeated_assets=(epochs - 1) * len(rows["train"]),
            consumed_slot_decisions=epochs * len(rows["train"]) * 6,
        )
        final_check = predict(model, values["train"] if smoke else values["validation"])
        other = TemporalSlotHead().eval()
        other.load_state_dict(torch.load(output / "final.pt", weights_only=True), strict=True)
        if not np.array_equal(
            final_check, predict(other, values["train"] if smoke else values["validation"])
        ):
            raise ValueError("final fresh head reload differs")
        if not smoke:
            report["final_raw"] = store_prediction(
                output, "final-raw", final_check, targets["validation"], rows["validation"]
            )
            final_train = predict(model, values["train"])
            report["best_epoch"] = selected_epoch(report["history"])
            report["selected_consumed_assets"] = report["best_epoch"] * len(rows["train"])
            report["selected_consumed_slot_decisions"] = report["selected_consumed_assets"] * 6
            model.load_state_dict(torch.load(output / "best.pt", weights_only=True), strict=True)
            cal_logits = predict(model, values["calibration"])
            temperatures = select_temperatures(cal_logits, targets["calibration"])
            report["temperatures"] = temperatures.tolist()
            report["calibration_evaluated"] = True
            store_prediction(
                output, "calibration", cal_logits, targets["calibration"], rows["calibration"]
            )
            best_logits = predict(model, values["validation"])
            report["best_raw"] = store_prediction(
                output, "best-raw", best_logits, targets["validation"], rows["validation"]
            )
            report["best_calibrated"] = store_prediction(
                output,
                "best-calibrated",
                best_logits,
                targets["validation"],
                rows["validation"],
                temperatures,
            )
            other.load_state_dict(torch.load(output / "best.pt", weights_only=True), strict=True)
            if not np.array_equal(best_logits, predict(other, values["validation"])):
                raise ValueError("best fresh head reload differs")
            with np.load(
                output / f"epoch-{report['best_epoch']:02}.npz", allow_pickle=False
            ) as snapshot:
                if not np.array_equal(best_logits, snapshot["logits"]):
                    raise ValueError("selected epoch prediction/reload differs")
            store_prediction(
                output,
                "best-train",
                predict(model, values["train"]),
                targets["train"],
                rows["train"],
            )
            store_prediction(output, "final-train", final_train, targets["train"], rows["train"])
        decision = json.loads(json_decision(final_check[0]))
        from scripts.lombard_readout_core import OPTIONS

        reverse = {s: list(reversed(o)) for s, o in OPTIONS.items()}
        if decision != json.loads(json_decision(final_check[0], reverse)):
            raise ValueError("native option permutation differs")
        report.update(
            status="PASS_TRAIN_ONLY_SMOKE" if smoke else "PASS_ARM_EXECUTION",
            final_reload_exact=True,
            json_option_permutation_exact=True,
            final_sha256=sha(output / "final.pt"),
            normalizer_sha256=sha(output / "normalizer.npz"),
        )
        if not smoke:
            report["best_sha256"] = sha(output / "best.pt")
        report["artifact_hashes"] = {
            p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "result.json"
        }
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        steps.close()
        try:
            report.update(**guard(run, tick))
        except Exception as error:
            report.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] == "FAIL_INCOMPLETE":
        raise RuntimeError("temporal arm failed resource finalization")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--smoke-sha256")
    args = parser.parse_args()
    if not args.smoke:
        raise ValueError(
            "this revision admits train-only smoke; full execution requires a separate "
            "reviewed controller, controls, bootstrap and fresh-process evaluator"
        )
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    controller = dict(
        status="RUNNING",
        arms={},
        smoke=args.smoke,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        device="cpu",
        dtype="float32",
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        gpu_used=False,
        encoders_loaded=False,
        cloud_cost=0,
        sealed_audit_opened=False,
        command=subprocess.list2cmdline(
            ["python", "-m", "scripts.train_lombard_temporal"] + sys.argv[1:]
        ),
    )
    try:
        guard(args.output, tick)
        pins = bound_sources()
        recheck(pins)
        config = read_json(CONFIG, pins[str(CONFIG.resolve())])
        required = dict(
            arms=ARMS,
            device="cpu",
            epochs=24,
            seed=17,
            batch_size=128,
            learning_rate=0.001,
            weight_decay=0.001,
            head_parameters=106355,
            frames=16,
            threads=4,
            dtype="float32",
            max_body_seconds=7200,
            max_peak_cpu_bytes=3 * 2**30,
            max_output_bytes=128 * 2**20,
            minimum_disk_bytes=2**30,
            full_source_counts=[4080, 398, 889],
        )
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError("fixed temporal comparison config differs")
        if not args.smoke:
            if args.smoke_reference is None or args.smoke_sha256 is None:
                raise ValueError(
                    "verified train-only smoke reference and SHA required before full run"
                )
            reference = read_json(args.smoke_reference / "result.json", args.smoke_sha256)
            admit_smoke(reference, pins)
            recheck(reference["validated_hashes"])
        controller.update(config=config, validated_hashes=pins)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        controller.update(
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )
        raw, framed, targets, rows, provenance = source_scope(args.smoke)
        if args.smoke:
            controller["timing_probe"] = timing_probe(raw, framed, fit_normalizer(*raw["train"]))
            if controller["timing_probe"]["projected_full_seconds"] >= 7200:
                raise ValueError("measured full CPU runtime does not fit fixed budget")
        controller["provenance"] = provenance
        (args.output / "effective-config.json").write_text(
            json.dumps(controller, indent=2) + "\n", encoding="utf-8"
        )
        controller["effective_config_sha256"] = sha(args.output / "effective-config.json")
        for arm in ARMS:
            controller["arms"][arm] = fit_arm(
                raw,
                framed,
                targets,
                rows,
                config,
                arm,
                args.output / arm,
                args.output,
                tick,
                args.smoke,
            )
            save_progress(args.output / "progress.json", controller)
        expected_status = "PASS_TRAIN_ONLY_SMOKE" if args.smoke else "PASS_ARM_EXECUTION"
        if any(r["status"] != expected_status for r in controller["arms"].values()):
            raise ValueError("all temporal arms must complete before PASS")
        if (
            len({r["initial_tensor_sha256"] for r in controller["arms"].values()}) != 1
            or len({r["sampler_sha256"] for r in controller["arms"].values()}) != 1
        ):
            raise ValueError("different head initialization or sample order")
        recheck(pins)
        if sha(args.output / "effective-config.json") != controller["effective_config_sha256"]:
            raise ValueError("effective runtime config changed")
        if source_scope(args.smoke)[4] != provenance:
            raise ValueError("frozen source/frame/cache authority changed")
        if torch.cuda.is_initialized():
            raise ValueError("CPU-only comparison unexpectedly initialized CUDA")
        controller["status"] = (
            "PASS_FIVE_ARM_TRAIN_ONLY_SMOKE" if args.smoke else "PASS_FIVE_ARM_TRAINING"
        )
    except Exception:
        controller.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        try:
            controller.update(**guard(args.output, tick))
        except Exception as error:
            controller.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        controller.update(ended_utc=datetime.now(UTC).isoformat())
        (args.output / "result.json").write_text(
            json.dumps(controller, indent=2) + "\n", encoding="utf-8"
        )
    if controller["status"] == "FAIL_INCOMPLETE":
        raise RuntimeError("temporal controller failed resource finalization")
    print(controller["status"], flush=True)


if __name__ == "__main__":
    main()
