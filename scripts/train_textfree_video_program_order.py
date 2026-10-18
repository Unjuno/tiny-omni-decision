"""Run a fixed-budget BiLSTM readout comparison with/without program positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.train_textfree_video_probe import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    _load_row_metrics,
    _paired_scene_bootstrap,
    _read_rows,
    _run_variant,
    _validate_inputs,
)
from tiny_omni_decision.textfree_video import build_video_vocab, encode_video_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_features(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    sample_dir = Path(config["source"]["sample_dir"]).resolve(strict=True)
    with np.load(
        Path(config["features"]["cache"]).resolve(strict=True), allow_pickle=False
    ) as data:
        hashes = data["media_sha256"].astype(str).tolist()
        values = data["features"]
        features = {media: values[index].copy() for index, media in enumerate(hashes)}
    return (
        features,
        _read_rows(sample_dir / "train.jsonl"),
        _read_rows(sample_dir / "validation.jsonl"),
    )


def run(config_path: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_path = config_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = _validate_inputs(config_path, config)
    device = _device(device_name)
    comparison = config["comparison"]
    if (
        comparison.get("seed") != SEED
        or comparison.get("epochs") != EPOCHS
        or comparison.get("batch_size") != BATCH_SIZE
        or comparison.get("learning_rate") != LEARNING_RATE
        or comparison.get("max_optimizer_updates") != 2048
        or comparison.get("early_stopping") is not False
    ):
        raise ValueError("program-order comparison must use the frozen seed-17 2,048-update budget")
    arms = comparison.get("arms")
    if arms != [
        {"id": "bilstm_mean_program", "temporal_mode": "bilstm", "program_pooling": "mean"},
        {
            "id": "bilstm_position_program",
            "temporal_mode": "bilstm",
            "program_pooling": "sinusoidal_position",
        },
    ]:
        raise ValueError("comparison arms must match the preregistered program-order pair")
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(device)

    output_dir.mkdir(parents=True, exist_ok=False)
    effective = {
        **config,
        "resolved": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "repository_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": [sys.executable, *sys.argv],
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "trainer_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "scripts" / "train_textfree_video_probe.py"
            ),
            "model_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "tiny_omni_decision"
                / "textfree_video.py"
            ),
        },
    }
    effective_path = output_dir / "effective-config.json"
    effective_path.write_text(
        json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight_path = output_dir / "input-preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    features, train_rows, validation_rows = _move_features(config)
    vocab = build_video_vocab(train_rows)
    train = encode_video_rows(train_rows, features, vocab)
    validation = encode_video_rows(validation_rows, features, vocab)
    epoch_orders = [
        np.random.default_rng(SEED + epoch).permutation(len(train_rows)) for epoch in range(EPOCHS)
    ]
    started = datetime.now(UTC)
    started_counter = time.perf_counter()
    runs = {}
    for arm in arms:
        arm_id = str(arm["id"])
        runs[arm_id] = _run_variant(
            arm_id,
            vocab,
            train,
            validation,
            train_rows,
            validation_rows,
            output_dir,
            device,
            epoch_orders,
            temporal_mode=str(arm["temporal_mode"]),
            program_pooling=str(arm["program_pooling"]),
        )

    left_id, right_id = (str(arms[1]["id"]), str(arms[0]["id"]))
    left = _load_row_metrics(output_dir / f"selected-validation-{left_id}.jsonl")
    right = _load_row_metrics(output_dir / f"selected-validation-{right_id}.jsonl")
    paired = {
        f"{question_type}/{metric}": _paired_scene_bootstrap(
            left, right, question_type, metric, left_id, right_id
        )
        for question_type in ("temporal_descriptive", "static_descriptive")
        for metric in ("accuracy", "nll", "brier")
    }
    report = {
        "status": "PASS_TEXTFREE_VIDEO_PROGRAM_ORDER_FIT",
        "experiment_id": config["id"],
        "source_commit": effective["resolved"]["repository_commit"],
        "config_sha256": _sha256(config_path),
        "effective_config_sha256": _sha256(effective_path),
        "input_preflight_sha256": _sha256(preflight_path),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started_counter,
        "command": [sys.executable, *sys.argv],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_examples": len(train_rows),
        "train_unique_scenes": len({str(row["scene_group_id"]) for row in train_rows}),
        "validation_examples": len(validation_rows),
        "validation_unique_scenes": len({str(row["scene_group_id"]) for row in validation_rows}),
        "updates_per_arm": 2048,
        "sealed_audit_loaded": False,
        "validation_used_for_checkpoint_selection": True,
        "runs": runs,
        "paired_scene_bootstrap_position_minus_mean": paired,
        "bootstrap": {
            "unit": "scene_group_id",
            "repetitions": 5000,
            "seed": SEED,
            "interval": "percentile 95%",
            "validation_selection_effect": "development validation also selects each best epoch",
        },
        "cloud_cost": 0,
    }
    report_path = output_dir / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run-report.sha256").write_text(_sha256(report_path) + "\n", encoding="ascii")
    return report


def main() -> None:
    args = _args()
    print(json.dumps(run(args.config, args.output_dir, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
