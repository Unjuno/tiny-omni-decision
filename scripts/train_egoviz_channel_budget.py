"""Run the preregistered matched 512-epoch frozen-feature comparison on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.egoviz_channel_budget_policy import evaluation_epochs, improves

EPOCHS = 512
FIXED = {32, 64, 128, 256, 384, 512}


def run(corpus: Path, baseline_report: Path, output: Path):
    import torch

    from scripts.egoviz_channel_scorer import ChannelWeightedScorer
    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
        metrics,
        sha256_file,
    )

    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3_000_000_000:
        raise RuntimeError("less than 3 GB disk reserve")
    report = json.loads(baseline_report.read_text(encoding="utf-8"))
    files = {
        "train_samples": corpus / "train-samples.jsonl",
        "development_samples": corpus / "development-samples.jsonl",
        "video_features": corpus / "video-features.npz",
        "sensor_features": corpus / "sensor-features.npz",
    }
    if report["sealed_audit_accessed"] is not False:
        raise ValueError("baseline does not attest to audit isolation")
    if {key: sha256_file(path) for key, path in files.items()} != report["input_sha256"]:
        raise ValueError("frozen input hash mismatch")
    torch.set_num_threads(4)
    rows, video, sensor = _load_samples(corpus, files["video_features"], files["sensor_features"])
    train = _tensor_data(rows, video, sensor, "train")
    dev = _tensor_data(rows, video, sensor, "development")
    if len(train[4]) != 332 or len(dev[4]) != 307:
        raise ValueError("unexpected train or development count")
    if (
        len(np.unique(train[3])) != 24
        or len(np.unique(dev[3])) != 20
        or set(train[3]) & set(dev[3])
    ):
        raise ValueError("unexpected or overlapping session groups")
    models = {}
    for name, factory in (
        ("baseline", JointTemporalScorer),
        ("channel_weighted", ChannelWeightedScorer),
    ):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        models[name] = factory().cpu()
    for name in ("video_projection.weight", "sensor_projection.weight"):
        if not torch.equal(
            models["baseline"].state_dict()[name], models["channel_weighted"].state_dict()[name]
        ):
            raise ValueError("initial projection weights differ")
    with torch.inference_mode():
        baseline_initial = models["baseline"](train[0][:8], train[1][:8])
        candidate_initial = models["channel_weighted"](train[0][:8], train[1][:8])
    initial_delta = float((baseline_initial - candidate_initial).abs().max())
    if initial_delta > 1e-6:
        raise ValueError("initial gate differs from uniform mean")
    output.mkdir(parents=True)
    status_path = output / "status.json"
    status = {
        "status": "RUNNING",
        "stage": "initializing",
        "epoch": 0,
        "sealed_audit_accessed": False,
        "initial_logit_max_difference": initial_delta,
    }

    def save_status():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save_status()
    results = {}
    order_hashes = {}
    epochs_to_evaluate = set(evaluation_epochs(EPOCHS))
    try:
        for name, model in models.items():
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            generator = torch.Generator().manual_seed(17)
            order_hash = hashlib.sha256()
            history = []
            best_nll = math.inf
            best_epoch = 0
            best_state = None
            best_train_metrics = None
            best_dev_metrics = None
            started = time.perf_counter()
            updates = 0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = torch.randperm(len(train[2]), generator=generator)
                order_hash.update(order.numpy().astype("<i4").tobytes())
                total_loss = 0.0
                last_grad_norm = 0.0
                for batch in order.split(32):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train[0][batch], train[1][batch])
                    loss = torch.nn.functional.cross_entropy(logits, train[2][batch])
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite loss in {name} epoch {epoch}")
                    loss.backward()
                    squared = 0.0
                    for parameter in model.parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise ValueError(
                                f"missing or nonfinite gradient in {name} epoch {epoch}"
                            )
                        squared += float(parameter.grad.detach().square().sum())
                    last_grad_norm = math.sqrt(squared)
                    optimizer.step()
                    updates += 1
                    total_loss += float(loss.detach()) * len(batch)
                if epoch not in epochs_to_evaluate:
                    continue
                model.eval()
                with torch.inference_mode():
                    train_logits = model(train[0], train[1]).numpy()
                    dev_logits = model(dev[0], dev[1]).numpy()
                if not np.isfinite(train_logits).all() or not np.isfinite(dev_logits).all():
                    raise ValueError(f"nonfinite evaluation in {name} epoch {epoch}")
                train_metrics = metrics(train_logits, train[2].numpy())
                dev_metrics = metrics(dev_logits, dev[2].numpy())
                history.append(
                    {
                        "epoch": epoch,
                        "updates": updates,
                        "rolling_train_ce": total_loss / len(train[2]),
                        "train": train_metrics,
                        "development": dev_metrics,
                        "last_batch_gradient_norm": last_grad_norm,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if improves(dev_metrics["nll"], best_nll):
                    best_nll, best_epoch = dev_metrics["nll"], epoch
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    best_train_metrics, best_dev_metrics = train_metrics, dev_metrics
                    torch.save(best_state, output / f"{name}-best.pt")
                if epoch in FIXED:
                    torch.save(model.state_dict(), output / f"{name}-epoch-{epoch}.pt")
                    rows_out = []
                    probs = torch.softmax(torch.from_numpy(dev_logits), dim=1).numpy()
                    for index, sample_id in enumerate(dev[4]):
                        rows_out.append(
                            {
                                "sample_id": sample_id,
                                "group_id": str(dev[3][index]),
                                "target": int(dev[2][index]),
                                "logits": dev_logits[index].tolist(),
                                "probabilities": probs[index].tolist(),
                            }
                        )
                    (output / f"{name}-epoch-{epoch}-predictions.jsonl").write_text(
                        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_out),
                        encoding="utf-8",
                    )
                status.update(
                    stage=name,
                    epoch=epoch,
                    updates=updates,
                    best_epoch=best_epoch,
                    best_nll=best_nll,
                    elapsed_seconds=time.perf_counter() - started,
                )
                save_status()
            if best_state is None:
                raise ValueError("no best checkpoint selected")
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                best_logits = model(dev[0], dev[1]).numpy()
                permuted = model(dev[0], dev[1][:, [2, 0, 3, 1]]).numpy()
            np.testing.assert_allclose(permuted, best_logits[:, [2, 0, 3, 1]], atol=1e-6, rtol=1e-5)
            best_probs = torch.softmax(torch.from_numpy(best_logits), dim=1).numpy()
            rows_out = [
                {
                    "sample_id": sample_id,
                    "group_id": str(dev[3][index]),
                    "target": int(dev[2][index]),
                    "logits": best_logits[index].tolist(),
                    "probabilities": best_probs[index].tolist(),
                }
                for index, sample_id in enumerate(dev[4])
            ]
            (output / f"{name}-best-predictions.jsonl").write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_out),
                encoding="utf-8",
            )
            order_hashes[name] = order_hash.hexdigest()
            results[name] = {
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "best_epoch": best_epoch,
                "best_train": best_train_metrics,
                "best_development": best_dev_metrics,
                "history": history,
                "total_updates": updates,
                "unique_train_examples": len(train[2]),
                "sample_exposures": EPOCHS * len(train[2]),
                "best_checkpoint_exposures": best_epoch * len(train[2]),
                "elapsed_seconds": time.perf_counter() - started,
                "sample_order_sha256": order_hashes[name],
                "best_checkpoint_sha256": sha256_file(output / f"{name}-best.pt"),
                "final_checkpoint_sha256": sha256_file(output / f"{name}-epoch-512.pt"),
                "option_permutation": "PASS",
            }
        if order_hashes["baseline"] != order_hashes["channel_weighted"]:
            raise ValueError("training order differs between arms")
        payload = {
            "status": "FIXED_BUDGET_COMPARISON_COMPLETE_EXPLORATORY",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runner_sha256": sha256_file(Path(__file__)),
            "input_sha256": report["input_sha256"],
            "initial_logit_max_difference": initial_delta,
            "seed": 17,
            "max_epochs": EPOCHS,
            "batch_size": 32,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "cpu_threads": 4,
            "selection": "development NLL, min_delta 1e-6, no early stop",
            "results": results,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "sealed_audit_accessed": False,
        }
        (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        status.update(status="COMPLETE", stage="done", epoch=EPOCHS)
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
        raise
    finally:
        save_status()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.baseline_report, args.output), indent=2))


if __name__ == "__main__":
    main()
