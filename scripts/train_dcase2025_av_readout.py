"""Fit fixed-budget audio-only, video-only and joint frozen-feature readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.dcase2025_readout import DCASE2025Readout, binary_probability_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_one(mode, audio, video, event_class, targets, *, seed, epochs, batch_size, device):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = DCASE2025Readout(mode, hidden_width=64).to(device)
    x_audio = torch.tensor(audio, dtype=torch.float32, device=device)
    x_video = torch.tensor(video, dtype=torch.float32, device=device)
    x_event = torch.as_tensor(event_class, dtype=torch.long, device=device)
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    mean_audio = x_audio.mean(0) if mode in {"audio", "joint"} else torch.zeros(768, device=device)
    std_audio = (
        x_audio.std(0, unbiased=False).clamp_min(1e-6)
        if mode in {"audio", "joint"}
        else torch.ones(768, device=device)
    )
    mean_video = x_video.mean(0) if mode in {"video", "joint"} else torch.zeros(768, device=device)
    std_video = (
        x_video.std(0, unbiased=False).clamp_min(1e-6)
        if mode in {"video", "joint"}
        else torch.ones(768, device=device)
    )
    x_audio = (x_audio - mean_audio) / std_audio
    x_video = (x_video - mean_video) / std_video
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    steps = math.ceil(len(y) / batch_size)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator).to(device)
        loss_total = 0.0
        correct = 0
        for start in range(0, len(y), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_audio[indices], x_video[indices], x_event[indices])
            loss = F.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {mode} epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(-1) == y[indices]).sum())
        history.append(
            {
                "epoch": epoch + 1,
                "train_ce": loss_total / len(y),
                "rolling_train_accuracy": correct / len(y),
                "optimizer_updates": (epoch + 1) * steps,
            }
        )
    model.eval()
    with torch.inference_mode():
        probabilities = model(x_audio, x_video, x_event).softmax(-1).cpu().numpy()
    state = {
        f"model.{key}": value.detach().cpu().numpy() for key, value in model.state_dict().items()
    }
    state.update(
        mean_audio=mean_audio.cpu().numpy(),
        scale_audio=std_audio.cpu().numpy(),
        mean_video=mean_video.cpu().numpy(),
        scale_video=std_video.cpu().numpy(),
    )
    return state, probabilities, history, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = args.features
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "PASS" or progress.get("valid_rows") != progress.get(
        "total_queries"
    ):
        raise ValueError("feature run is not complete and fully valid")
    rows = [
        json.loads(line)
        for line in (root / "query-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != progress["total_queries"] or len({row["sample_id"] for row in rows}) != len(
        rows
    ):
        raise ValueError("feature/query row integrity check failed")
    audio = np.load(root / "features_audio.npy", mmap_mode="r")
    video = np.load(root / "features_video.npy", mmap_mode="r")
    valid = np.load(root / "valid_mask.npy", mmap_mode="r")
    if audio.shape != (len(rows), 768) or video.shape != audio.shape or not valid.all():
        raise ValueError("feature tensor shape or validity check failed")
    event_class = np.asarray([row["event_class"] for row in rows], dtype=np.int64)
    targets = np.asarray([row["target_onscreen"] for row in rows], dtype=np.int64)
    if (event_class < 0).any() or (event_class >= 13).any() or not np.isin(targets, (0, 1)).all():
        raise ValueError("unexpected task labels")
    args.output_dir.mkdir(parents=True)
    config = {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "readout_module_sha256": sha256(
            Path(__file__).resolve().parents[1]
            / "src"
            / "tiny_omni_decision"
            / "dcase2025_readout.py"
        ),
        "source_feature_dir": str(root.resolve()),
        "feature_run_identity": progress["identity"],
        "feature_audio_sha256": sha256(root / "features_audio.npy"),
        "feature_video_sha256": sha256(root / "features_video.npy"),
        "query_manifest_sha256": sha256(root / "query-manifest.jsonl"),
        "row_order_sha256": progress["identity"]["query_ids_sha256"],
        "rows": len(rows),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "optimizer": "AdamW",
        "loss": "unweighted cross entropy",
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_count": torch.get_num_threads(),
        "modes": ["audio", "video", "joint"],
        "hidden_width": 64,
        "event_class_encoding": "13-way one-hot query context",
        "fixed_options": ["offscreen", "onscreen"],
        "checkpoint_rule": "fixed final epoch; no validation or early stopping",
        "source_split": "DCASE dev-train only; no validation",
    }
    (args.output_dir / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "status": "RUNNING",
        "config": config,
        "modes": {},
        "train_targets": {
            "offscreen": int((targets == 0).sum()),
            "onscreen": int((targets == 1).sum()),
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    try:
        for mode in ("audio", "video", "joint"):
            state, probabilities, history, elapsed = fit_one(
                mode,
                audio,
                video,
                event_class,
                targets,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
            checkpoint = args.output_dir / f"{mode}-readout.npz"
            np.savez_compressed(checkpoint, **state)
            prediction_path = args.output_dir / f"{mode}-train-predictions.npz"
            np.savez_compressed(
                prediction_path,
                sample_ids=np.asarray([r["sample_id"] for r in rows]),
                targets=targets,
                probabilities=probabilities,
            )
            history_path = args.output_dir / f"{mode}-history.json"
            history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            result["modes"][mode] = {
                "metrics_on_training_rows_not_validation": binary_probability_metrics(
                    probabilities, targets
                ),
                "elapsed_seconds": elapsed,
                "optimizer_updates": args.epochs * math.ceil(len(rows) / args.batch_size),
                "trainable_parameters": sum(
                    value.size for key, value in state.items() if key.startswith("model.")
                ),
                "checkpoint_sha256": sha256(checkpoint),
                "predictions_sha256": sha256(prediction_path),
                "history_sha256": sha256(history_path),
            }
            (args.output_dir / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
