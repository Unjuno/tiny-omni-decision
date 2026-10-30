"""Fit one fixed-budget GRU over frozen EgoViz video/sensor features."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.train_egoviz_vs_sync_readout import (
    _load_samples,
    _tensor_data,
    metrics,
    sha256_file,
)

SEED = 17
UPDATES = 5632
EVALUATION_STEPS = (352, 704, 1408, 2816, 4224, 5632)


class OrderedCrossModalGRUScorer(nn.Module):
    """Score each option by recurrently reading aligned video/sensor tokens."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.video_projection = nn.Linear(768, hidden_size)
        self.sensor_projection = nn.Linear(512, hidden_size)
        self.pair_projection = nn.Linear(hidden_size * 4, hidden_size)
        self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, video: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if video.ndim != 3 or tuple(video.shape[1:]) != (8, 768):
            raise ValueError("video must have shape [batch, 8, 768]")
        if candidates.ndim != 5 or tuple(candidates.shape[1:]) != (4, 6, 16, 512):
            raise ValueError("candidates must have shape [batch, 4, 6, 16, 512]")

        batch = video.shape[0]
        video_tokens = nn.functional.normalize(self.video_projection(video), dim=-1)
        sensor_tokens = candidates.mean(dim=2).reshape(batch, 4, 8, 2, 512).mean(dim=3)
        sensor_tokens = nn.functional.normalize(self.sensor_projection(sensor_tokens), dim=-1)
        video_tokens = video_tokens[:, None].expand(-1, 4, -1, -1)
        paired = torch.cat(
            (
                video_tokens,
                sensor_tokens,
                video_tokens * sensor_tokens,
                torch.abs(video_tokens - sensor_tokens),
            ),
            dim=-1,
        )
        paired = torch.tanh(self.pair_projection(paired)).reshape(batch * 4, 8, -1)
        _, final_state = self.recurrent(paired)
        return self.score(final_state[-1]).reshape(batch, 4)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_predictions(path: Path, sample_ids, groups, targets):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(sample_ids):
        raise ValueError(f"baseline prediction count mismatch: {path}")
    for index, row in enumerate(rows):
        if (
            row["sample_id"] != sample_ids[index]
            or row["group_id"] != str(groups[index])
            or int(row["target"]) != int(targets[index])
        ):
            raise ValueError(f"baseline prediction identity mismatch at row {index}: {path}")
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    if logits.shape != (len(sample_ids), 4) or not np.isfinite(logits).all():
        raise ValueError(f"invalid baseline logits: {path}")
    return logits


def _paired_group_bootstrap(candidate_logits, baseline_logits, targets, groups):
    unique = np.unique(groups)
    rng = np.random.default_rng(SEED)
    draws = {key: [] for key in ("accuracy", "nll", "brier")}
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        candidate = metrics(candidate_logits[indices], targets[indices])
        baseline = metrics(baseline_logits[indices], targets[indices])
        for key in draws:
            draws[key].append(candidate[key] - baseline[key])
    point_candidate = metrics(candidate_logits, targets)
    point_baseline = metrics(baseline_logits, targets)
    return {
        "method": "paired session bootstrap; 2,000 percentile draws; seed 17",
        "group_count": int(len(unique)),
        "candidate_minus_linear": {
            key: {
                "estimate": point_candidate[key] - point_baseline[key],
                "percentile_95_ci": [
                    float(np.quantile(draws[key], 0.025)),
                    float(np.quantile(draws[key], 0.975)),
                ],
            }
            for key in draws
        },
    }


def run(corpus: Path, baseline_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if shutil.disk_usage(output.parent).free < 3_000_000_000:
        raise RuntimeError("less than 3,000,000,000 bytes free disk reserve")
    merge_path = corpus / "merge-report.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if (
        merge.get("status") != "VALIDATED_TRAIN_EXPANSION_FIXED_DEVELOPMENT"
        or merge.get("sealed_audit_accessed") is not False
        or merge.get("development_video_and_sensor_features_unchanged") is not True
    ):
        raise ValueError("merged-corpus integrity gate failed")
    for name, digest in merge["sha256"].items():
        if sha256_file(corpus / name) != digest:
            raise ValueError(f"frozen corpus hash mismatch: {name}")

    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows, video_by_id, sensor_features = _load_samples(
        corpus, corpus / "video-features.npz", corpus / "sensor-features.npz"
    )
    train = _tensor_data(rows, video_by_id, sensor_features, "train")
    dev = _tensor_data(rows, video_by_id, sensor_features, "development")
    if (
        len(train[4]) != 773
        or len(dev[4]) != 307
        or len(np.unique(train[3])) != 48
        or len(np.unique(dev[3])) != 20
        or set(train[3]) & set(dev[3])
    ):
        raise ValueError("frozen train/development split mismatch")

    model = OrderedCrossModalGRUScorer().cpu()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 123_521:
        raise ValueError(f"unexpected trainable readout parameter count: {parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SEED)
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "step": 0,
        "sealed_audit_accessed": False,
    }
    _write_json(output / "status.json", status)
    started = time.perf_counter()
    history = []
    sample_exposures = 0
    order_hash = hashlib.sha256()
    steps_per_pass = (len(train[2]) + 31) // 32
    step = 0
    while step < UPDATES:
        order = torch.randperm(len(train[2]), generator=generator)
        for indices in order.split(32):
            if step >= UPDATES:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(train[0][indices], train[1][indices])
            loss = nn.functional.cross_entropy(logits, train[2][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {step + 1}")
            loss.backward()
            grad_squared = 0.0
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"missing/nonfinite gradient at update {step + 1}")
                grad_squared += float(parameter.grad.detach().square().sum())
            optimizer.step()
            order_hash.update(indices.numpy().astype("<i4").tobytes())
            sample_exposures += len(indices)
            step += 1
            if step not in EVALUATION_STEPS:
                continue

            model.eval()
            with torch.inference_mode():
                train_logits = model(train[0], train[1]).numpy()
                dev_logits = model(dev[0], dev[1]).numpy()
            if not np.isfinite(train_logits).all() or not np.isfinite(dev_logits).all():
                raise FloatingPointError(f"nonfinite evaluation at update {step}")
            name = f"step-{step}"
            state_path = output / f"{name}.pt"
            torch.save(model.state_dict(), state_path)
            probabilities = torch.softmax(torch.from_numpy(dev_logits), dim=1).numpy()
            pred_path = output / f"{name}-predictions.jsonl"
            pred_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "sample_id": dev[4][index],
                            "group_id": str(dev[3][index]),
                            "target": int(dev[2][index]),
                            "logits": dev_logits[index].tolist(),
                            "probabilities": probabilities[index].tolist(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                    for index in range(len(dev[4]))
                ),
                encoding="utf-8",
            )
            baseline_logits = _load_predictions(
                baseline_dir / f"step-{step}-predictions.jsonl", dev[4], dev[3], dev[2]
            )
            paired = _paired_group_bootstrap(dev_logits, baseline_logits, dev[2].numpy(), dev[3])
            history.append(
                {
                    "update": step,
                    "completed_passes": step // steps_per_pass,
                    "sample_exposures": sample_exposures,
                    "train": metrics(train_logits, train[2].numpy()),
                    "development": metrics(dev_logits, dev[2].numpy()),
                    "linear_reference_development": metrics(baseline_logits, dev[2].numpy()),
                    "paired_session_bootstrap": paired,
                    "checkpoint_sha256": sha256_file(state_path),
                    "prediction_sha256": sha256_file(pred_path),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            _write_json(output / "history.json", history)
            status.update({"step": step, "last_checkpoint": state_path.name})
            _write_json(output / "status.json", status)

    report = {
        "status": "FIXED_BUDGET_COMPLETE_EXPLORATORY_DEVELOPMENT",
        "source_commit": status["source_commit"],
        "runner_sha256": sha256_file(Path(__file__)),
        "started_at": status["started_at"],
        "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": "cpu",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "seed": SEED,
        "training": {
            "updates": UPDATES,
            "sample_exposures": sample_exposures,
            "train_examples": len(train[4]),
            "development_examples": len(dev[4]),
            "train_sessions": len(np.unique(train[3])),
            "development_sessions": len(np.unique(dev[3])),
            "parameter_count": parameters,
            "optimizer": "AdamW(lr=0.003, weight_decay=0.01)",
            "batch_size": 32,
            "cpu_threads": torch.get_num_threads(),
            "evaluation_steps": list(EVALUATION_STEPS),
            "early_stopping": False,
            "sample_order_sha256": order_hash.hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "architecture": (
            "video/sensor pair features -> shared 64D projection -> "
            "unidirectional GRU(64) -> shared scalar score"
        ),
        "input_sha256": {name: sha256_file(corpus / name) for name in merge["sha256"]},
        "merge_report_sha256": sha256_file(merge_path),
        "baseline_results_sha256": sha256_file(baseline_dir / "results.json"),
        "sealed_audit_accessed": False,
        "interpretation": "development-only; fixed endpoint is not a blind or product evaluation",
        "history": history,
    }
    _write_json(output / "results.json", report)
    status.update(
        {
            "status": report["status"],
            "step": UPDATES,
            "finished_at": report["finished_at"],
            "results_sha256": sha256_file(output / "results.json"),
        }
    )
    _write_json(output / "status.json", status)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.corpus, args.baseline_dir, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "updates": report["training"]["updates"],
                "elapsed_seconds": report["training"]["elapsed_seconds"],
                "parameter_count": report["training"]["parameter_count"],
                "final_development": report["history"][-1]["development"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
