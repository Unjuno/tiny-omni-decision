"""Fixed paired appearance comparison; complete audited corpus before fitting."""

import argparse
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
from scripts.cover_point_metrics import membership_metrics, state_hash
from scripts.extract_spatial_context_corpus import load_plans
from scripts.spatial_readout_inputs import SPATIAL_RESULT_PIN, context_tokens, load_spatial_bank
from scripts.train_role_supervised_qa import auxiliary_loss, load, qa_summary, verify_saved_epoch
from scripts.train_trajectory_readout import ROOT
from tiny_omni_decision.spatial_trajectory_readout import SpatialTrajectoryReadout


def require_limits(start, config):
    if time.perf_counter() - start > config["max_seconds"]:
        raise RuntimeError("predeclared one-hour cap exceeded; incomplete experiment")
    if torch.cuda.max_memory_allocated() > config["peak_cap_bytes"]:
        raise MemoryError("4GiB predeclared allocation cap exceeded")


def process_peak_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("process peak memory query failed")
        return counters.peak_working
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def paired_video_bootstrap(a, loga, b, logb, targets):
    y = np.asarray(targets)
    if len(y) % 2 or len(y) == 0:
        raise ValueError("nonempty two-question video pairs required")
    values = []
    for p, logp in ((a, loga), (b, logb)):
        if p.shape != (len(y), 3) or logp.shape != p.shape or not np.isfinite(logp).all():
            raise ValueError("aligned original QA probabilities/log probabilities required")
        values.append(
            dict(
                accuracy=(p.argmax(-1) == y).reshape(-1, 2).mean(1),
                nll=(-logp[np.arange(len(y)), y]).reshape(-1, 2).mean(1),
                brier=((p - np.eye(3)[y]) ** 2).sum(1).reshape(-1, 2).mean(1),
            )
        )
    draws = np.random.default_rng(17).integers(0, len(y) // 2, (10000, len(y) // 2))
    result = dict(
        repetitions=10000,
        seed=17,
        videos=len(y) // 2,
        unit="video ID",
        method="percentile",
        direction="temporal minus initial broadcast",
        limitation="selected on observed development; not blind or proven actor-disjoint",
    )
    for key in values[0]:
        delta = values[1][key] - values[0][key]
        result[key] = dict(
            difference=float(delta.mean()),
            interval95=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
        )
    return result


def paired_batch(bank, spatial, data, indices, arm, device, control="normal"):
    points, contexts, texts, targets = [], [], [], []
    for index in indices:
        vid = data["pairs"][index]["video_id"]
        context_index = (index + 1) % len(data["pairs"]) if control == "mismatch" else index
        context_vid = data["pairs"][context_index]["video_id"]
        value = bank[vid]
        points.append(torch.zeros_like(value) if control == "zero" else value)
        contexts.append(torch.from_numpy(context_tokens(spatial[context_vid], arm, control)))
        texts.append(data["text"][2 * index : 2 * index + 2].reshape(6, 1536))
        targets.append(data["targets"][2 * index : 2 * index + 2])
    return (
        torch.stack(points).to(device),
        torch.stack(texts).to(device),
        torch.stack(contexts).to(device),
        torch.cat(targets).to(device),
    )


def paired_logits(model, points, candidates, context):
    state = model.encode_state(points, context)
    return model.score_candidates(state, candidates).reshape(-1, 3), state[1]


def predict(model, bank, spatial, data, arm, device, control="normal", limits=None):
    qa, cover, logp = [], [], []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(data["pairs"]), 8):
            if limits is not None:
                require_limits(*limits)
            values = paired_batch(
                bank,
                spatial,
                data,
                range(offset, min(offset + 8, len(data["pairs"]))),
                arm,
                device,
                control,
            )
            points, candidates, context, _ = values
            state = model.encode_state(points, context)
            logits = model.score_candidates(state, candidates).reshape(-1, 3)
            reversed_candidates = candidates.reshape(-1, 2, 3, 1536).flip(2).reshape(-1, 6, 1536)
            reversed_logits = model.score_candidates(state, reversed_candidates).reshape(-1, 3)
            torch.testing.assert_close(reversed_logits, logits.flip(1), rtol=1e-5, atol=1e-6)
            qa.append(logits.softmax(-1).cpu().numpy())
            cover.append(state[1].sigmoid().cpu().numpy())
            logp.append(logits.log_softmax(-1).cpu().numpy())
            if limits is not None:
                require_limits(*limits)
    return np.concatenate(qa), np.concatenate(cover), np.concatenate(logp)


def training_state(model, optimizer, sampler, epoch, updates, config_hash):
    return dict(
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        updates=updates,
        sampler_rng=sampler.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        config_sha256=config_hash,
        model_sha256=state_hash(model),
        strict_resume_claimed=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        smoke=args.smoke,
        training=True,
        device="cuda",
        cloud_cost=0,
        sealed_audit_opened=False,
        encoders_loaded_for_training=False,
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
        if shutil.disk_usage(args.output).free < 8 * 1024**3:
            raise OSError("8GiB disk reserve mandatory")
        _, plans = load_plans()
        spatial, hashes = load_spatial_bank(
            ROOT / "spatial-context-corpus-01/result.json", plans, SPATIAL_RESULT_PIN
        )
        bank, data, other_hashes, missing = load()
        hashes.update(other_hashes)
        if set(bank) != set(spatial):
            raise ValueError("spatial/point identities differ")
        # Independent full-file/readback audit is persisted before any model initialization.
        write(
            args.output / "preflight-audit.json",
            dict(
                status="PASS_COMPLETE_CORPUS_AND_BINDINGS",
                videos=len(spatial),
                qa_counts={s: len(d["targets"]) for s, d in data.items()},
                file_hashes=hashes,
            ),
        )
        config = dict(
            seed=17,
            epochs=50,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_video_pairs=8,
            hidden=64,
            latent_count=16,
            heads=4,
            auxiliary_weight=1.0,
            appearance_arms=["initial_broadcast", "temporal_appearance"],
            selector="earliest minimum validation original-question stable log-softmax NLL",
            input_hashes=hashes,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            code_hashes={
                p: sha(Path(p))
                for p in [
                    "scripts/train_spatial_trajectory_readout.py",
                    "scripts/spatial_readout_inputs.py",
                    "scripts/train_role_supervised_qa.py",
                    "scripts/train_trajectory_readout.py",
                    "scripts/cover_point_metrics.py",
                    "src/tiny_omni_decision/spatial_trajectory_readout.py",
                    "src/tiny_omni_decision/cover_point_readout.py",
                ]
            },
            smoke=args.smoke,
            label_exceptions=missing,
            torch=torch.__version__,
            python=platform.python_version(),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            device="cuda",
            precision="FP32",
            tf32=False,
            max_seconds=3600,
            peak_cap_bytes=4 * 1024**3,
            frozen_encoders_loaded=False,
            threads=4,
        )
        write(args.output / "config.json", config)
        config_hash = sha(args.output / "config.json")
        result["config_sha256"] = config_hash
        for split, d in data.items():
            np.savez_compressed(
                args.output / f"{split}-labels.npz",
                cover=d["cover_targets"].numpy(),
                available=d["cover_available"].numpy(),
                qa=d["targets"].numpy(),
                identities=np.array([p["video_id"] for p in d["pairs"]]),
            )
        for arm in config["appearance_arms"]:
            folder = args.output / arm
            folder.mkdir()
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            sampler = torch.Generator().manual_seed(17)
            model = SpatialTrajectoryReadout().cuda()
            initial_sha = state_hash(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            torch.save(model.state_dict(), folder / "untrained.pt")
            best, best_epoch, updates, history, orders = float("inf"), None, 0, [], []
            train_seconds, evaluation_seconds = 0.0, 0.0
            arm_start = time.perf_counter()
            for epoch in range(1, 2 if args.smoke else 51):
                require_limits(start, config)
                order = torch.randperm(len(data["train"]["pairs"]), generator=sampler).tolist()
                consumed_order = []
                model.train()
                torch.cuda.synchronize()
                train_start = time.perf_counter()
                for offset in range(0, len(order), 8):
                    require_limits(start, config)
                    ids = order[offset : offset + 8]
                    points, candidates, context, target = paired_batch(
                        bank, spatial, data["train"], ids, arm, "cuda"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    logits, roles = paired_logits(model, points, candidates, context)
                    labels = data["train"]["cover_targets"][ids].cuda()
                    available = data["train"]["cover_available"][ids].cuda()
                    loss = F.cross_entropy(logits, target) + auxiliary_loss(
                        roles, labels, available
                    )
                    loss.backward()
                    if not torch.isfinite(loss) or any(
                        p.grad is None or not p.grad.isfinite().all() for p in model.parameters()
                    ):
                        raise ValueError("nonfinite/missing gradient or loss")
                    optimizer.step()
                    require_limits(start, config)
                    updates += 1
                    consumed_order.extend(ids)
                    if args.smoke and updates == 2:
                        break
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - train_start
                orders.append(consumed_order)
                eval_start = time.perf_counter()
                measurements, predictions, replay = {}, {}, {}
                splits = ("train",) if args.smoke else ("train", "valid")
                for split in splits:
                    p, c, logp = predict(
                        model, bank, spatial, data[split], arm, "cuda", limits=(start, config)
                    )
                    replay[split] = (p, c, logp)
                    predictions[split], predictions[split + "_cover"] = p, c
                    predictions[split + "_logp"] = logp
                    measurements[split] = qa_summary(p, data[split]["targets"].numpy(), logp)
                    known = data[split]["cover_available"].numpy()
                    measurements[split + "_cover_supplemental"] = membership_metrics(
                        c[known], data[split]["cover_targets"].numpy()[known]
                    )
                evaluation_seconds += time.perf_counter() - eval_start
                np.savez_compressed(folder / f"epoch-{epoch:02d}-predictions.npz", **predictions)
                score = measurements["train" if args.smoke else "valid"]["nll"]
                if score < best:
                    best, best_epoch = score, epoch
                    torch.save(model.state_dict(), folder / "best.pt")
                    torch.save(
                        training_state(model, optimizer, sampler, epoch, updates, config_hash),
                        folder / "best-training-state.pt",
                    )
                history.append(
                    dict(
                        epoch=epoch,
                        updates=updates,
                        best_epoch=best_epoch,
                        **measurements,
                        seconds=time.perf_counter() - arm_start,
                        train_seconds=train_seconds,
                        evaluation_seconds=evaluation_seconds,
                    )
                )
                write(folder / "history.json", history)
                result["progress"] = dict(
                    arm=arm, epoch=epoch, updates=updates, best_epoch=best_epoch, latest_nll=score
                )
                write(args.output / "progress.json", result)
                print(arm, epoch, "nll", score, "best", best_epoch, flush=True)
            torch.save(model.state_dict(), folder / "final.pt")
            torch.save(
                training_state(model, optimizer, sampler, epoch, updates, config_hash),
                folder / "final-training-state.pt",
            )
            write(folder / "orders.json", orders)
            assert state_hash(model) != initial_sha
            restored = SpatialTrajectoryReadout().cuda()
            for name, selected_epoch in (("final", epoch), ("best", best_epoch)):
                restored.load_state_dict(torch.load(folder / f"{name}.pt", weights_only=True))
                verify_saved_epoch(
                    folder / f"epoch-{selected_epoch:02d}-predictions.npz",
                    {
                        s: predict(
                            restored, bank, spatial, data[s], arm, "cuda", limits=(start, config)
                        )
                        for s in splits
                    },
                )
            controls, saved = {}, {}
            if not args.smoke:
                for control in ("normal", "zero", "mismatch", "initial", "reverse"):
                    p, c, logp = predict(
                        restored,
                        bank,
                        spatial,
                        data["valid"],
                        arm,
                        "cuda",
                        control,
                        limits=(start, config),
                    )
                    saved[control], saved[control + "_cover"] = p, c
                    saved[control + "_logp"] = logp
                    controls[control] = qa_summary(p, data["valid"]["targets"].numpy(), logp)
                np.savez_compressed(folder / "selected-controls.npz", **saved)
                verify_saved_epoch(
                    folder / f"epoch-{best_epoch:02d}-predictions.npz",
                    {"valid": (saved["normal"], saved["normal_cover"], saved["normal_logp"])},
                )
            presentations = sum(map(len, orders))
            selected_presentations = sum(map(len, orders[:best_epoch]))
            result["arms"][arm] = dict(
                best_epoch=best_epoch,
                final_epoch=epoch,
                updates=updates,
                initial_sha256=initial_sha,
                orders_sha256=sha(folder / "orders.json"),
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                run_video_presentations=presentations,
                selected_video_presentations=selected_presentations,
                unique_train_videos=len(set(i for order in orders for i in order)),
                run_question_presentations=2 * presentations,
                selected_question_presentations=2 * selected_presentations,
                controls=controls,
                train_seconds=train_seconds,
                evaluation_seconds=evaluation_seconds,
                seconds=time.perf_counter() - arm_start,
                reload_bitwise=True,
                hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()},
                best_weight_bytes=(folder / "best.pt").stat().st_size,
            )
            del model, restored, optimizer
            torch.cuda.empty_cache()
        a, b = list(result["arms"].values())
        if a["initial_sha256"] != b["initial_sha256"] or a["orders_sha256"] != b["orders_sha256"]:
            raise ValueError("paired initialization/sampling diverged")
        if not args.smoke:
            with (
                np.load(args.output / "initial_broadcast/selected-controls.npz") as a,
                np.load(args.output / "temporal_appearance/selected-controls.npz") as b,
            ):
                result["paired_video_bootstrap"] = paired_video_bootstrap(
                    a["normal"],
                    a["normal_logp"],
                    b["normal"],
                    b["normal_logp"],
                    data["valid"]["targets"].numpy(),
                )
        for path, pin in hashes.items():
            require_limits(start, config)
            if sha(Path(path)) != pin:
                raise ValueError("protected input changed during training")
        peak_process = process_peak_bytes()
        require_limits(start, config)
        result.update(
            status="PASS_TRAIN_ONLY_SMOKE"
            if args.smoke
            else "PASS_FIXED_SPATIAL_APPEARANCE_COMPARISON_ONLY",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_working_set_bytes=peak_process,
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - start, ended_utc=datetime.now(UTC).isoformat())
        write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
