"""Fixed train-scene coarse/fine QA probe, smoke-gated and separately replayable."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_correspondence_validation import sha, write
from scripts.cover_point_metrics import state_hash
from scripts.fine_qa_bank import load
from scripts.train_role_supervised_qa import auxiliary_loss, qa_summary, verify_saved_epoch
from scripts.train_spatial_trajectory_readout import paired_video_bootstrap, process_peak_bytes
from tiny_omni_decision.fine_qa_inputs import context_view
from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout

ARMS = ("coarse16", "fine32")
CODE = [
    "scripts/train_fine_grouped_qa.py",
    "scripts/fine_qa_bank.py",
    "src/tiny_omni_decision/fine_qa_inputs.py",
    "src/tiny_omni_decision/spatial_trajectory_readout.py",
    "src/tiny_omni_decision/cover_point_readout.py",
    "scripts/train_role_supervised_qa.py",
    "scripts/train_trajectory_readout.py",
    "scripts/train_spatial_trajectory_readout.py",
    "scripts/cover_point_metrics.py",
    "scripts/audit_correspondence_validation.py",
    "src/tiny_omni_decision/decision_math.py",
    "docs/pretrained_reuse/R2_FINE_QA_GROUPED_PROTOCOL_20261014.md",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def epoch_order(sampler, count):
    return torch.randperm(count, generator=sampler).tolist()


def evidence_pins(folder, result, mode):
    config_path, split_path = folder / "config.json", folder / "split.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        sha(config_path) != result["config_sha256"]
        or config["contract"] != result["contract"]
        or config["mode"] != mode
        or sha(split_path) != result["contract"]["split_sha256"]
        or digest(result["contract"]) != result["contract_sha256"]
    ):
        raise ValueError("actual reference config/mode/split does not bind evidence")
    pins = {
        str(folder / "result.json"): sha(folder / "result.json"),
        str(config_path): result["config_sha256"],
        str(split_path): result["contract"]["split_sha256"],
    }
    for name, arm in result["arms"].items():
        for file, pin in arm["hashes"].items():
            path = folder / name / file
            if sha(path) != pin:
                raise ValueError("actual reference artifact changed")
            pins[str(path)] = pin
    return pins


def verify_smoke(folder, contract):
    result = json.loads((folder / "result.json").read_text())
    if (
        result["status"] != "PASS_TRAIN_ONLY_SMOKE"
        or result["contract"] != contract
        or result["contract_sha256"] != digest(contract)
        or not 0 < result["projected_main_seconds"] < 3600
        or set(result["arms"]) != set(ARMS)
    ):
        raise ValueError("actual smoke contract/projection not admitted")
    a, b = [result["arms"][name] for name in ARMS]
    if (
        a["initial_sha256"] != b["initial_sha256"]
        or a["prefix_orders"] != b["prefix_orders"]
        or any(r["parameters"] != 232385 for r in (a, b))
    ):
        raise ValueError("smoke initialization/order/parameter mismatch")
    result["_reference_pins"] = evidence_pins(folder, result, "smoke")
    return result


def batch(points, contexts, data, indices, arm, device, control="normal"):
    p, c, t, y = [], [], [], []
    for i in indices:
        donor = (i + 1) % len(data["pairs"]) if control == "mismatch" else i
        vid = data["pairs"][donor]["video_id"]
        p.append(
            torch.from_numpy(points[vid])
            if control != "zero"
            else torch.zeros_like(torch.from_numpy(points[vid]))
        )
        a, pos = contexts[arm][vid]
        if isinstance(a, str):
            a, pos = (
                np.load(a, mmap_mode="r", allow_pickle=False),
                np.load(pos, mmap_mode="r", allow_pickle=False),
            )
        c.append(
            torch.from_numpy(context_view(a, pos, "normal" if control == "mismatch" else control))
        )
        t.append(torch.from_numpy(data["text"][2 * i : 2 * i + 2].reshape(6, 1536)))
        y.append(torch.from_numpy(data["targets"][2 * i : 2 * i + 2]))
    return (
        torch.stack(p).to(device),
        torch.stack(t).to(device),
        torch.stack(c).to(device),
        torch.cat(y).to(device),
    )


def limits(start, output):
    if time.perf_counter() - start > 3600:
        raise RuntimeError("one-hour body cap exceeded; incomplete")
    if torch.cuda.max_memory_allocated() > 4 * 1024**3 or process_peak_bytes() > 4 * 1024**3:
        raise MemoryError("4GiB GPU/process allocation cap exceeded")
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise OSError("8GiB disk reserve exhausted")


def predict(model, points, contexts, data, arm, start, output, control="normal"):
    qa, cover, logp = [], [], []
    model.eval()
    with torch.no_grad():
        for off in range(0, len(data["pairs"]), 8):
            limits(start, output)
            p, t, c, _ = batch(
                points,
                contexts,
                data,
                range(off, min(off + 8, len(data["pairs"]))),
                arm,
                "cuda",
                control,
            )
            state = model.encode_state(p, c)
            logits = model.score_candidates(state, t).reshape(-1, 3)
            rev = t.reshape(-1, 2, 3, 1536).flip(2).reshape(-1, 6, 1536)
            reversed_logits = model.score_candidates(state, rev).reshape(-1, 3)
            torch.testing.assert_close(reversed_logits, logits.flip(1), rtol=1e-5, atol=1e-6)
            if not torch.equal(reversed_logits.flip(1).argmax(-1), logits.argmax(-1)):
                raise ValueError("reversed candidates changed original answer identity")
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite QA scores")
            qa.append(logits.softmax(-1).cpu().numpy())
            logp.append(logits.log_softmax(-1).cpu().numpy())
            cover.append(state[1].sigmoid().cpu().numpy())
            limits(start, output)
    return np.concatenate(qa), np.concatenate(cover), np.concatenate(logp)


def measurements(values, data):
    p, _, logp = values
    result = qa_summary(p, data["targets"], logp)
    result["mean_confidence"] = float(p.max(1).mean())
    result["pair_count"] = len(data["pairs"])
    result["both_correct_count"] = int(
        ((p.argmax(-1) == data["targets"]).reshape(-1, 2)).all(1).sum()
    )
    return result


def flatten_predictions(values):
    return {
        split + suffix: value
        for split, v in values.items()
        for suffix, value in zip(("", "_cover", "_logp"), v, strict=True)
    }


def subset(data, indices):
    rows = [q for i in indices for q in (2 * i, 2 * i + 1)]
    return dict(
        pairs=[data["pairs"][i] for i in indices],
        text=data["text"][rows],
        targets=data["targets"][rows],
        cover=data["cover"][indices],
        available=data["available"][indices],
    )


def train_state(model, optimizer, sampler, epoch, updates, config_hash):
    return dict(
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        updates=updates,
        sampler_rng=sampler.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        model_sha256=state_hash(model),
        config_sha256=config_hash,
        strict_resume_claimed=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--smoke-from", type=Path)
    group.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        training=not bool(args.replay_from),
        smoke=args.smoke,
        device="cuda",
        cloud_cost=0,
        official_validation_arrays_or_labels_opened=False,
        sealed_audit_opened=False,
        encoders_loaded=False,
        arms={},
    )
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            raise MemoryError("12GiB free GPU preflight mandatory")
        limits(start, args.output)
        points, contexts, data, hashes, split = load()
        write(args.output / "split.json", split)
        contract = dict(
            seed=17,
            epochs=50,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_video_pairs=8,
            auxiliary_weight=1,
            model="unchanged SpatialTrajectoryReadout64/16/4",
            parameters=232385,
            arms=list(ARMS),
            selector="earliest strictly lowest selector stable original-question NLL",
            primary_comparison="matched final epoch50",
            input_hashes=hashes,
            split_sha256=sha(args.output / "split.json"),
            code_hashes={p: sha(Path(p)) for p in CODE},
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dtype="float32",
            tf32=False,
            deterministic=True,
            torch=torch.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            threads=4,
            sdpa=dict(
                flash=torch.backends.cuda.flash_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            ),
            max_seconds=3600,
            peak_gpu_process_bytes=4 * 1024**3,
            disk_reserve_bytes=8 * 1024**3,
            cached_text_revision="4ca70771034acceecb2e72475f72050fcdde4ddc",
            cached_tapir_revision="730cda1c730877cfedbe01bf87fb1cadb78a565d",
            logical_feature_plus_head_parameters=48694536,
            strict_resume_claimed=False,
        )
        result.update(contract=contract, contract_sha256=digest(contract))
        write(
            args.output / "config.json",
            dict(
                contract=contract,
                mode="smoke" if args.smoke else "replay" if args.replay_from else "main",
                command=subprocess.list2cmdline(
                    ["python", "-m", "scripts.train_fine_grouped_qa"] + os.sys.argv[1:]
                ),
            ),
        )
        config_hash = sha(args.output / "config.json")
        result["config_sha256"] = config_hash
        write(
            args.output / "preflight-audit.json",
            dict(
                status="PASS_TRAIN_ONLY_NATIVE_BINDINGS",
                file_hashes=hashes,
                split_sha256=contract["split_sha256"],
                fit_videos=79,
                selector_videos=20,
            ),
        )
        for name, d in data.items():
            np.savez_compressed(
                args.output / f"{name}-labels.npz",
                qa=d["targets"],
                cover=d["cover"],
                available=d["available"],
                identities=np.array([p["video_id"] for p in d["pairs"]]),
            )
        smoke = verify_smoke(args.smoke_from, contract) if args.smoke_from else None
        references = smoke["_reference_pins"] if smoke else {}
        if args.replay_from:
            original = json.loads((args.replay_from / "result.json").read_text())
            if (
                original["status"] != "PASS_FIXED_GROUPED_FINE_QA_COMPARISON_ONLY"
                or original["contract"] != contract
            ):
                raise ValueError("fresh replay contract changed")
            references = evidence_pins(args.replay_from, original, "main")
            for arm in ARMS:
                folder = args.replay_from / arm
                for file, pin in original["arms"][arm]["hashes"].items():
                    if sha(folder / file) != pin:
                        raise ValueError("original replay artifact changed")
                restored = SpatialTrajectoryReadout().cuda()
                for label in ("best", "final"):
                    restored.load_state_dict(torch.load(folder / f"{label}.pt", weights_only=True))
                    epoch = original["arms"][arm]["best_epoch"] if label == "best" else 50
                    replay = {
                        s: predict(restored, points, contexts, d, arm, start, args.output)
                        for s, d in data.items()
                    }
                    verify_saved_epoch(folder / f"epoch-{epoch:02d}-predictions.npz", replay)
                    np.savez_compressed(
                        args.output / f"{arm}-{label}-replay.npz", **flatten_predictions(replay)
                    )
                result["arms"][arm] = dict(
                    best_final_original_predictions_bitwise=True,
                    hashes={p.name: sha(p) for p in args.output.glob(f"{arm}-*-replay.npz")},
                )
                del restored
                torch.cuda.empty_cache()
            result["status"] = "PASS_FRESH_ALL_FOUR_WEIGHT_REPLAYS_NOT_QUALITY"
        else:
            projected = 0
            for arm in ARMS:
                folder = args.output / arm
                folder.mkdir()
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                sampler = torch.Generator().manual_seed(17)
                model = SpatialTrajectoryReadout().cuda()
                initial = state_hash(model)
                if sum(p.numel() for p in model.parameters()) != 232385:
                    raise ValueError("immutable model parameter count changed")
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                torch.save(model.state_dict(), folder / "untrained.pt")
                updates, orders, history, best, best_epoch = 0, [], [], float("inf"), None
                train_seconds, eval_seconds = 0.0, 0.0
                arm_start = time.perf_counter()
                prefix = None
                eval_data = data
                for epoch in range(1, 2 if args.smoke else 51):
                    order = epoch_order(sampler, 79)
                    if args.smoke:
                        order = order[:16]
                        eval_data = {"fit": subset(data["fit"], order)}
                    orders.append(order)
                    ce_sum, aux_sum, seen, gradients = 0.0, 0.0, 0, []
                    model.train()
                    tick = time.perf_counter()
                    for offset in range(0, len(order), 8):
                        limits(start, args.output)
                        ids = order[offset : offset + 8]
                        p, t, c, y = batch(points, contexts, data["fit"], ids, arm, "cuda")
                        optimizer.zero_grad(set_to_none=True)
                        state = model.encode_state(p, c)
                        logits = model.score_candidates(state, t).reshape(-1, 3)
                        ce = F.cross_entropy(logits, y)
                        aux = auxiliary_loss(
                            state[1],
                            torch.from_numpy(data["fit"]["cover"][ids]).cuda(),
                            torch.from_numpy(data["fit"]["available"][ids]).cuda(),
                        )
                        loss = ce + aux
                        loss.backward()
                        if not torch.isfinite(loss) or any(
                            p.grad is None or not p.grad.isfinite().all()
                            for p in model.parameters()
                        ):
                            raise ValueError("nonfinite/missing gradient or loss")
                        gradients.append(
                            float(
                                torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
                            )
                        )
                        optimizer.step()
                        updates += 1
                        if any(not p.isfinite().all() for p in model.parameters()):
                            raise ValueError("nonfinite updated parameter")
                        ce_sum += float(ce) * len(ids)
                        aux_sum += float(aux) * len(ids)
                        seen += len(ids)
                        if updates == 2:
                            prefix = state_hash(model)
                            if smoke and prefix != smoke["arms"][arm]["prefix_state_sha256"]:
                                raise ValueError(
                                    "main first two updates differ from separate smoke"
                                )
                        limits(start, args.output)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    train_seconds += elapsed
                    tick = time.perf_counter()
                    values = {
                        s: predict(model, points, contexts, d, arm, start, args.output)
                        for s, d in eval_data.items()
                    }
                    stats = {s: measurements(values[s], d) for s, d in eval_data.items()}
                    torch.cuda.synchronize()
                    evaluation = time.perf_counter() - tick
                    eval_seconds += evaluation
                    np.savez_compressed(
                        folder / f"epoch-{epoch:02d}-predictions.npz", **flatten_predictions(values)
                    )
                    score = stats["fit" if args.smoke else "selector"]["nll"]
                    if score < best:
                        best, best_epoch = score, epoch
                        torch.save(model.state_dict(), folder / "best.pt")
                        torch.save(
                            train_state(model, optimizer, sampler, epoch, updates, config_hash),
                            folder / "best-training-state.pt",
                        )
                    history.append(
                        dict(
                            epoch=epoch,
                            updates=updates,
                            best_epoch=best_epoch,
                            **stats,
                            train_ce=ce_sum / seen,
                            train_cover_aux=aux_sum / seen,
                            gradient_norms=gradients,
                            lr=0.001,
                            cumulative_video_presentations=sum(map(len, orders)),
                            cumulative_question_presentations=2 * sum(map(len, orders)),
                            unique_video_assets=len(set(i for o in orders for i in o)),
                            training_seconds=elapsed,
                            evaluation_seconds=evaluation,
                            seconds=time.perf_counter() - arm_start,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        )
                    )
                    write(folder / "history.json", history)
                    print(
                        arm,
                        epoch,
                        "fit",
                        stats["fit"]["accuracy"],
                        "selector",
                        stats.get("selector"),
                        flush=True,
                    )
                torch.save(model.state_dict(), folder / "final.pt")
                torch.save(
                    train_state(model, optimizer, sampler, epoch, updates, config_hash),
                    folder / "final-training-state.pt",
                )
                write(folder / "orders.json", orders)
                restored = SpatialTrajectoryReadout().cuda()
                for label, selected_epoch in [("final", epoch), ("best", best_epoch)]:
                    restored.load_state_dict(torch.load(folder / f"{label}.pt", weights_only=True))
                    replay = {
                        s: predict(restored, points, contexts, d, arm, start, args.output)
                        for s, d in eval_data.items()
                    }
                    verify_saved_epoch(
                        folder / f"epoch-{selected_epoch:02d}-predictions.npz", replay
                    )
                controls = {}
                if not args.smoke:
                    saved = {}
                    for control in ("normal", "zero", "mismatch", "initial", "reverse"):
                        v = predict(
                            restored,
                            points,
                            contexts,
                            data["selector"],
                            arm,
                            start,
                            args.output,
                            control,
                        )
                        saved.update(flatten_predictions({control: v}))
                        controls[control] = measurements(v, data["selector"])
                    np.savez_compressed(folder / "selected-controls.npz", **saved)
                    verify_saved_epoch(
                        folder / f"epoch-{best_epoch:02d}-predictions.npz",
                        {
                            "selector": (
                                saved["normal"],
                                saved["normal_cover"],
                                saved["normal_logp"],
                            )
                        },
                    )
                presentations = sum(map(len, orders))
                selected_presentations = sum(map(len, orders[:best_epoch]))
                result["arms"][arm] = dict(
                    parameters=232385,
                    trainable_parameters=232385,
                    initial_sha256=initial,
                    prefix_orders=orders[0][:16],
                    prefix_state_sha256=prefix,
                    best_epoch=best_epoch,
                    final_epoch=epoch,
                    updates=updates,
                    run_video_presentations=presentations,
                    selected_video_presentations=selected_presentations,
                    run_question_presentations=2 * presentations,
                    selected_question_presentations=2 * selected_presentations,
                    unique_train_video_assets=len(set(i for o in orders for i in o)),
                    repeated_video_presentations=presentations
                    - len(set(i for o in orders for i in o)),
                    train_seconds=train_seconds,
                    evaluation_seconds=eval_seconds,
                    seconds=time.perf_counter() - arm_start,
                    controls=controls,
                    reload_bitwise=True,
                    hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()},
                )
                if args.smoke:
                    # Measured16-video evaluation includes two candidate orderings;
                    # main evaluates99 per epoch + replays/controls. Add 20% margin.
                    projected += (train_seconds / 2 * 500 + eval_seconds / 16 * 99 * 50) * 1.2
                del model, restored, optimizer
                torch.cuda.empty_cache()
            a, b = [result["arms"][name] for name in ARMS]
            if (
                a["initial_sha256"] != b["initial_sha256"]
                or a["prefix_orders"] != b["prefix_orders"]
                or a["hashes"]["orders.json"] != b["hashes"]["orders.json"]
            ):
                raise ValueError("matched initialization or full sampling diverged")
            if args.smoke:
                result["projected_main_seconds"] = projected
                if projected >= 3600:
                    raise RuntimeError("main projected beyond fixed hour cap")
                result["status"] = "PASS_TRAIN_ONLY_SMOKE"
            else:
                bootstrap = {}
                for label in ("final", "selected"):
                    values = []
                    for arm in ARMS:
                        ep = 50 if label == "final" else result["arms"][arm]["best_epoch"]
                        with np.load(
                            args.output / arm / f"epoch-{ep:02d}-predictions.npz",
                            allow_pickle=False,
                        ) as v:
                            values.extend([v["selector"].copy(), v["selector_logp"].copy()])
                    record = paired_video_bootstrap(*values, data["selector"]["targets"])
                    record["direction"] = "fine32 minus coarse16"
                    record["limitation"] = (
                        "known train scenes; selector checkpoint selection; "
                        "not blind or actor-disjoint"
                    )
                    bootstrap[label] = record
                result["paired_scene_bootstrap"] = bootstrap
                result["status"] = "PASS_FIXED_GROUPED_FINE_QA_COMPARISON_ONLY"
        # Integrity is checked again before recording PASS, including executable code.
        result["reference_evidence_hashes"] = references
        for path, pin in {**hashes, **contract["code_hashes"], **references}.items():
            limits(start, args.output)
            if sha(Path(path)) != pin:
                raise ValueError("protected input/executable changed during run")
        if (
            sha(args.output / "config.json") != config_hash
            or sha(args.output / "split.json") != contract["split_sha256"]
        ):
            raise ValueError("effective config/split changed")
        limits(start, args.output)
        result.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_working_set_bytes=process_peak_bytes(),
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - start, ended_utc=datetime.now(UTC).isoformat())
        write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
