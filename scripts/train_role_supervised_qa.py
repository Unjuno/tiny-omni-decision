"""Paired QA comparison: identical model/budget with role auxiliary weight0/1."""

import argparse
import json
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_correspondence_validation import sha, write
from scripts.cover_point_metrics import arm_input, membership_metrics, state_hash
from scripts.train_trajectory_readout import ROOT, batch, load_inputs, summarize
from tiny_omni_decision.cover_point_readout import (
    balanced_membership_loss,
    initial_cover_membership,
)
from tiny_omni_decision.occlusion_pairs import build_pairs
from tiny_omni_decision.supervised_trajectory_readout import SupervisedTrajectoryReadout


def auxiliary_loss(logits, labels, available):
    if available.shape != logits.shape[:1] or available.dtype != torch.bool:
        raise ValueError("aligned Boolean availability required")
    if not available.any():
        return logits.sum() * 0
    return balanced_membership_loss(logits[available], labels[available])


def qa_summary(probabilities, targets, log_probabilities):
    """Original-question metrics with stable, unfloored checkpoint-selection NLL."""
    p, y, logp = np.asarray(probabilities), np.asarray(targets), np.asarray(log_probabilities)
    if logp.shape != p.shape or not np.isfinite(logp).all():
        raise ValueError("aligned finite log probabilities required")
    result = summarize(p, y)
    losses = -logp[np.arange(len(y)), y].astype(np.float64)
    result["nll"] = float(losses.mean())
    result["initial"]["nll"] = float(losses[::2].mean())
    result["final"]["nll"] = float(losses[1::2].mean())
    changed = y.reshape(-1, 2)[:, 0] != y.reshape(-1, 2)[:, 1]
    for name, mask in (("changed", changed), ("unchanged", ~changed)):
        if mask.any():
            result[name]["nll"] = float(losses[np.repeat(mask, 2)].mean())
    return result


def verify_saved_epoch(path, predictions):
    """Compare replay to original inference outputs, not another loaded model."""
    with np.load(path, allow_pickle=False) as original:
        for split, values in predictions.items():
            for suffix, replay in zip(("", "_cover", "_logp"), values, strict=True):
                np.testing.assert_array_equal(original[split + suffix], replay)


def load():
    bank, data, hashes = load_inputs(
        ROOT / "perception-occlusion-trajectory-features-01",
        ROOT / "perception-occlusion-text-features-01",
    )
    xy = np.array([[(x + 0.5) / 16, (y + 0.5) / 16] for y in range(16) for x in range(16)])
    sources = {
        "train": (
            ROOT / "perception-test-source-audit-01/train_annotations.zip",
            "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78",
        ),
        "valid": (
            ROOT / "perception-correspondence-valid-audit-01/valid_annotations.zip",
            "eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7",
        ),
    }
    missing = {}
    for split, (path, pin) in sources.items():
        if sha(path) != pin:
            raise ValueError("source changed")
        hashes[str(path)] = pin
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != [f"all_{split}.json"]:
                raise ValueError("unexpected archive")
            source = json.loads(archive.read(f"all_{split}.json"))
        if build_pairs(source, split)["pairs"] != data[split]["pairs"]:
            raise ValueError("source corpus/answer remapping mismatch")
        labels, available, exceptions = [], [], []
        for pair in data[split]["pairs"]:
            try:
                target = initial_cover_membership(source[pair["video_id"]], xy)
                known = True
            except ValueError as error:
                if str(error) not in {
                    "one grounded cover required",
                    "one frame-zero cover box required",
                }:
                    raise
                target, known = np.zeros(256, dtype=bool), False
                exceptions.append(
                    dict(video_id=pair["video_id"], reason=str(error), qa_retained=True)
                )
            labels.append(target)
            available.append(known)
        data[split]["cover_targets"] = torch.from_numpy(np.stack(labels))
        data[split]["cover_available"] = torch.tensor(available)
        missing[split] = exceptions
        if sum(available) != {"train": 98, "valid": 253}[split]:
            raise ValueError("predeclared auxiliary coverage changed")
        del source
    return bank, data, hashes, missing


def predict(model, bank, data, control="normal"):
    qa, cover, logp = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data["pairs"]), 8):
            ids = range(i, min(i + 8, len(data["pairs"])))
            v, text, mask, _ = batch(
                bank, data, ids, control if control != "temporal_zero" else "normal"
            )
            if control == "temporal_zero":
                v = arm_input(v, "initial_only")
            logits, roles = model(v, text, mask)
            shuffled, _ = model(v, text.flip(1), mask)
            torch.testing.assert_close(shuffled, logits.flip(1))
            qa.append(logits.softmax(-1).numpy())
            logp.append(logits.log_softmax(-1).numpy())
            cover.append(roles[::2].sigmoid().numpy())
    return np.concatenate(qa), np.concatenate(cover), np.concatenate(logp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        smoke=args.smoke,
        training=True,
        device="cpu",
        gpu_time=0,
        cloud_cost=0,
        sealed_audit_opened=False,
        arms={},
    )
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        bank, data, hashes, missing = load()
        config = dict(
            seed=17,
            epochs=50,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_video_pairs=8,
            hidden=64,
            selector="earliest minimum validation original-question NLL",
            auxiliary_weights=[0.0, 1.0],
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            helper_sha256=sha(Path("scripts/cover_point_metrics.py")),
            nll_calculation="mean negative target log-softmax; no probability floor",
            input_hashes=hashes,
            modules={
                name: sha(Path("src/tiny_omni_decision") / name)
                for name in ["cover_point_readout.py", "supervised_trajectory_readout.py"]
            },
            label_exceptions=missing,
            qa_counts={s: len(d["targets"]) for s, d in data.items()},
            torch=torch.__version__,
            device="cpu",
            threads=4,
            smoke=args.smoke,
        )
        write(args.output / "config.json", config)
        result["config_sha256"] = sha(args.output / "config.json")
        for split, d in data.items():
            np.savez_compressed(
                args.output / f"{split}-labels.npz",
                cover=d["cover_targets"].numpy(),
                available=d["cover_available"].numpy(),
                qa=d["targets"].numpy(),
                identities=np.array([p["video_id"] for p in d["pairs"]]),
            )
        for weight in config["auxiliary_weights"]:
            name = "qa_only" if weight == 0 else "role_supervised"
            folder = args.output / name
            folder.mkdir()
            torch.manual_seed(17)
            sampler = torch.Generator().manual_seed(17)
            model = SupervisedTrajectoryReadout()
            initial_sha = state_hash(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            torch.save(model.state_dict(), folder / "untrained.pt")
            best, best_epoch, history, orders, updates = float("inf"), None, [], [], 0
            arm_start = time.perf_counter()
            for epoch in range(1, 2 if args.smoke else 51):
                order = torch.randperm(len(data["train"]["pairs"]), generator=sampler).tolist()
                orders.append(order)
                model.train()
                for offset in range(0, len(order), 8):
                    ids = order[offset : offset + 8]
                    v, text, mask, target = batch(bank, data["train"], ids)
                    optimizer.zero_grad(set_to_none=True)
                    logits, roles = model(v, text, mask)
                    loss = F.cross_entropy(logits, target)
                    if weight:
                        labels = data["train"]["cover_targets"][ids].repeat_interleave(2, 0)
                        available = data["train"]["cover_available"][ids].repeat_interleave(2)
                        loss = loss + weight * auxiliary_loss(roles, labels, available)
                    loss.backward()
                    if not torch.isfinite(loss) or any(
                        p.grad is None or not p.grad.isfinite().all() for p in model.parameters()
                    ):
                        raise ValueError("nonfinite/missing gradient or loss")
                    optimizer.step()
                    updates += 1
                    if args.smoke and updates == 2:
                        break
                measurements, probabilities = {}, {}
                for split in ("train",) if args.smoke else ("train", "valid"):
                    p, c, logp = predict(model, bank, data[split])
                    probabilities[split], probabilities[split + "_cover"] = p, c
                    probabilities[split + "_logp"] = logp
                    measurements[split] = qa_summary(p, data[split]["targets"].numpy(), logp)
                    known = data[split]["cover_available"].numpy()
                    measurements[split + "_cover_supplemental"] = membership_metrics(
                        c[known], data[split]["cover_targets"].numpy()[known]
                    )
                np.savez_compressed(folder / f"epoch-{epoch:02d}-predictions.npz", **probabilities)
                score = measurements["train" if args.smoke else "valid"]["nll"]
                if score < best:
                    best, best_epoch = score, epoch
                    torch.save(model.state_dict(), folder / "best.pt")
                    torch.save(
                        dict(
                            optimizer=optimizer.state_dict(),
                            epoch=epoch,
                            updates=updates,
                            sampler_rng=sampler.get_state(),
                            torch_rng=torch.get_rng_state(),
                            config_sha256=result["config_sha256"],
                        ),
                        folder / "best-training-state.pt",
                    )
                history.append(
                    dict(
                        epoch=epoch,
                        updates=updates,
                        best_epoch=best_epoch,
                        **measurements,
                        seconds=time.perf_counter() - arm_start,
                    )
                )
                write(folder / "history.json", history)
                print(name, epoch, "nll", score, "best", best_epoch, flush=True)
            assert state_hash(model) != initial_sha
            torch.save(model.state_dict(), folder / "final.pt")
            torch.save(
                dict(
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    updates=updates,
                    sampler_rng=sampler.get_state(),
                    torch_rng=torch.get_rng_state(),
                    config_sha256=result["config_sha256"],
                ),
                folder / "final-training-state.pt",
            )
            write(folder / "orders.json", orders)
            restored = SupervisedTrajectoryReadout()
            restored.load_state_dict(torch.load(folder / "final.pt", weights_only=True))
            replay_splits = ("train",) if args.smoke else ("train", "valid")
            verify_saved_epoch(
                folder / f"epoch-{epoch:02d}-predictions.npz",
                {split: predict(restored, bank, data[split]) for split in replay_splits},
            )
            model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
            restored.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
            verify_saved_epoch(
                folder / f"epoch-{best_epoch:02d}-predictions.npz",
                {split: predict(restored, bank, data[split]) for split in replay_splits},
            )
            controls, saved = {}, {}
            if not args.smoke:
                for control in ["normal", "zero", "mismatch", "temporal_zero"]:
                    p, c, logp = predict(model, bank, data["valid"], control)
                    saved[control], saved[control + "_cover"] = p, c
                    saved[control + "_logp"] = logp
                    controls[control] = qa_summary(p, data["valid"]["targets"].numpy(), logp)
                np.savez_compressed(folder / "selected-controls.npz", **saved)
                for a, b in zip(
                    predict(restored, bank, data["valid"]),
                    (saved["normal"], saved["normal_cover"], saved["normal_logp"]),
                    strict=True,
                ):
                    np.testing.assert_array_equal(a, b)
            result["arms"][name] = dict(
                auxiliary_weight=weight,
                best_epoch=best_epoch,
                final_epoch=epoch,
                updates=updates,
                initial_sha256=initial_sha,
                orders_sha256=sha(folder / "orders.json"),
                parameters=sum(p.numel() for p in model.parameters()),
                controls=controls,
                run_presentations=16 if args.smoke else epoch * 99,
                selected_presentations=16 if args.smoke else best_epoch * 99,
                seconds=time.perf_counter() - arm_start,
                reload_bitwise=True,
                hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()},
            )
        a, b = list(result["arms"].values())
        if a["initial_sha256"] != b["initial_sha256"] or a["orders_sha256"] != b["orders_sha256"]:
            raise ValueError("paired initialization/sampling diverged")
        if not args.smoke:
            truth = data["valid"]["targets"].numpy().reshape(-1, 2)
            hit_arrays = []
            for name in result["arms"]:
                with np.load(args.output / name / "selected-controls.npz") as saved:
                    hit_arrays.append((saved["normal"].argmax(-1).reshape(-1, 2) == truth).mean(1))
            delta = hit_arrays[1] - hit_arrays[0]
            draws = np.random.default_rng(17).integers(0, len(delta), (10000, len(delta)))
            result["paired_qa_bootstrap"] = dict(
                difference=float(delta.mean()),
                interval95=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
                repetitions=10000,
                seed=17,
                unit="video ID",
                method="percentile",
                limitation="selected checkpoints on observed development; actor isolation unproven",
            )
        result["status"] = (
            "PASS_TRAIN_ONLY_SMOKE"
            if args.smoke
            else "PASS_FIXED_ROLE_SUPERVISED_QA_COMPARISON_ONLY"
        )
    except Exception:
        result["status"], result["error"] = "FAIL_INCOMPLETE", traceback.format_exc()
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
