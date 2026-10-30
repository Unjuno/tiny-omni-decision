"""Train the fixed uniform readout on expanded sessions for 5,632 updates."""

from __future__ import annotations

import argparse
import datetime
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

from scripts.egoviz_channel_budget_policy import improves
from scripts.egoviz_coverage_budget_policy import budget_geometry, evaluation_steps

UPDATES = 5632
FIXED_STEPS = {352, 704, 1408, 2816, 4224, 5632}


def run(corpus: Path, baseline_report: Path, output: Path):
    import torch

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
        raise RuntimeError("less than 3 GB free disk reserve")
    merge_path = corpus / "merge-report.json"
    merged = json.loads(merge_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    if (
        merged["status"] != "VALIDATED_TRAIN_EXPANSION_FIXED_DEVELOPMENT"
        or merged["sealed_audit_accessed"] is not False
        or merged.get("development_video_and_sensor_features_unchanged") is not True
        or baseline["sealed_audit_accessed"] is not False
    ):
        raise ValueError("unverified split or audit isolation")
    for name, digest in merged["sha256"].items():
        if sha256_file(corpus / name) != digest:
            raise ValueError(f"merged input hash mismatch: {name}")
    if (
        merged["sha256"]["development-samples.jsonl"]
        != baseline["input_sha256"]["development_samples"]
    ):
        raise ValueError("development sample IDs/order changed")
    torch.set_num_threads(4)
    rows, video, sensor = _load_samples(
        corpus, corpus / "video-features.npz", corpus / "sensor-features.npz"
    )
    train = _tensor_data(rows, video, sensor, "train")
    dev = _tensor_data(rows, video, sensor, "development")
    if (
        len(train[4]) != merged["train_examples"]
        or len(dev[4]) != 307
        or len(np.unique(train[3])) != 48
        or len(np.unique(dev[3])) != 20
        or set(train[3]) & set(dev[3])
    ):
        raise ValueError("train or development split differs from verified merge")
    geometry = budget_geometry(examples=len(train[2]), batch_size=32, updates=UPDATES)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = JointTemporalScorer().cpu()
    init_hash = hashlib.sha256()
    for tensor in model.state_dict().values():
        init_hash.update(tensor.detach().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    generator = torch.Generator().manual_seed(17)
    sampled = np.zeros(len(train[2]), dtype=np.int64)
    selected_counts = None
    sample_order_hash = hashlib.sha256()
    output.mkdir(parents=True)
    started = time.perf_counter()
    status_path = output / "status.json"
    status = {
        "status": "RUNNING",
        "step": 0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "merge_report_sha256": sha256_file(merge_path),
        "sealed_audit_accessed": False,
    }

    def save_status():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def save_predictions(label, logits):
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        items = [
            {
                "sample_id": sample_id,
                "group_id": str(dev[3][index]),
                "target": int(dev[2][index]),
                "logits": logits[index].tolist(),
                "probabilities": probabilities[index].tolist(),
            }
            for index, sample_id in enumerate(dev[4])
        ]
        (output / f"{label}-predictions.jsonl").write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in items),
            encoding="utf-8",
        )

    save_status()
    history = []
    selected_state = None
    selected_train = None
    selected_dev = None
    selected_step = 0
    best_nll = math.inf
    step = 0
    passes_started = 0
    interval_loss = 0.0
    interval_examples = 0
    eval_steps = set(evaluation_steps(UPDATES))
    try:
        while step < UPDATES:
            passes_started += 1
            order = torch.randperm(len(train[2]), generator=generator)
            for batch in order.split(32):
                if step >= UPDATES:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(train[0][batch], train[1][batch])
                loss = torch.nn.functional.cross_entropy(logits, train[2][batch])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at update {step + 1}")
                loss.backward()
                squared = 0.0
                for parameter in model.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError(f"missing or nonfinite gradient at update {step + 1}")
                    squared += float(parameter.grad.detach().square().sum())
                grad_norm = math.sqrt(squared)
                optimizer.step()
                indices = batch.numpy()
                sampled[indices] += 1
                sample_order_hash.update(indices.astype("<i4").tobytes())
                step += 1
                interval_loss += float(loss.detach()) * len(batch)
                interval_examples += len(batch)
                if step not in eval_steps:
                    continue
                if shutil.disk_usage(output).free < 3_000_000_000:
                    raise RuntimeError("disk reserve breached during training")
                model.eval()
                with torch.inference_mode():
                    train_logits = model(train[0], train[1]).numpy()
                    dev_logits = model(dev[0], dev[1]).numpy()
                if not np.isfinite(train_logits).all() or not np.isfinite(dev_logits).all():
                    raise ValueError(f"nonfinite evaluation at update {step}")
                train_metrics = metrics(train_logits, train[2].numpy())
                dev_metrics = metrics(dev_logits, dev[2].numpy())
                history.append(
                    {
                        "update": step,
                        "pass_started": passes_started,
                        "interval_train_ce": interval_loss / interval_examples,
                        "train": train_metrics,
                        "development": dev_metrics,
                        "sample_exposures": int(sampled.sum()),
                        "unique_examples": int((sampled > 0).sum()),
                        "last_batch_gradient_norm": grad_norm,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                interval_loss = 0.0
                interval_examples = 0
                if improves(dev_metrics["nll"], best_nll):
                    best_nll, selected_step = dev_metrics["nll"], step
                    selected_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    selected_train, selected_dev = train_metrics, dev_metrics
                    selected_counts = sampled.copy()
                    torch.save(selected_state, output / "best.pt")
                if step in FIXED_STEPS:
                    torch.save(model.state_dict(), output / f"step-{step}.pt")
                    save_predictions(f"step-{step}", dev_logits)
                status.update(
                    step=step,
                    pass_started=passes_started,
                    sample_exposures=int(sampled.sum()),
                    best_step=selected_step,
                    best_nll=best_nll,
                    elapsed_seconds=time.perf_counter() - started,
                )
                save_status()
        if selected_state is None or selected_counts is None:
            raise ValueError("no selected checkpoint")
        if sampled.sum() != geometry["sample_exposures"]:
            raise ValueError("consumed examples differ from fixed budget accounting")
        model.load_state_dict(selected_state)
        model.eval()
        with torch.inference_mode():
            best_logits = model(dev[0], dev[1]).numpy()
            permuted = model(dev[0], dev[1][:, [2, 0, 3, 1]]).numpy()
        np.testing.assert_allclose(permuted, best_logits[:, [2, 0, 3, 1]], atol=1e-6, rtol=1e-5)
        save_predictions("best", best_logits)
        sample_rows = [row for row in rows if row["split"] == "train"]
        if [row["sample_id"] for row in sample_rows] != train[4]:
            raise ValueError("training IDs changed during accounting")
        with (output / "train-exposure-counts.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(sample_rows):
                item = {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "chunk_id": row["chunk_id"],
                    "category": row["category"],
                    "source": "original" if index < 332 else "expansion",
                    "full_run_exposures": int(sampled[index]),
                    "selected_checkpoint_exposures": int(selected_counts[index]),
                }
                stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        result = {
            "status": "FIXED_COVERAGE_BUDGET_COMPLETE_EXPLORATORY",
            "source_commit": status["source_commit"],
            "runner_sha256": sha256_file(Path(__file__)),
            "merge_report_sha256": sha256_file(merge_path),
            "input_sha256": merged["sha256"],
            "seed": 17,
            "device": "cpu",
            "cpu_threads": 4,
            "updates": step,
            "train_examples": len(train[2]),
            "original_examples": 332,
            "expansion_examples": len(train[2]) - 332,
            "development_examples": len(dev[2]),
            "train_sessions": 48,
            "development_sessions": 20,
            "geometry": geometry,
            "unique_consumed": int((sampled > 0).sum()),
            "sample_exposures": int(sampled.sum()),
            "repeated_example_exposures": int(sampled.sum() - (sampled > 0).sum()),
            "original_exposures": int(sampled[:332].sum()),
            "expansion_exposures": int(sampled[332:].sum()),
            "selected_checkpoint_exposures": int(selected_counts.sum()),
            "sample_order_sha256": sample_order_hash.hexdigest(),
            "initial_state_sha256": init_hash.hexdigest(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_step": selected_step,
            "best_train": selected_train,
            "best_development": selected_dev,
            "history": history,
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
            "final_checkpoint_sha256": sha256_file(output / "step-5632.pt"),
            "option_permutation": "PASS",
            "elapsed_seconds": time.perf_counter() - started,
            "started_at": status["started_at"],
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sealed_audit_accessed": False,
        }
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        status.update(status="COMPLETE", step=step)
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
        raise
    finally:
        save_status()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.baseline_report, args.output), indent=2))


if __name__ == "__main__":
    main()
