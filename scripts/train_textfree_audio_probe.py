"""Fit one CPU text-free readout on the existing frozen Speech Commands cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SOURCE = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"
AUDIO_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    count = len(targets)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=len(LABELS)).float()
    ece = 0.0
    for bin_index in range(15):
        mask = (confidence >= bin_index / 15) & (
            confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    return {
        "count": count,
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _load_rows(path: Path, expected_split: str, sample_dir: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(
        row.get("split") != expected_split or row.get("source") != SOURCE for row in rows
    ):
        raise ValueError(f"invalid {expected_split} sample rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate {expected_split} sample IDs")
    if any(row.get("target") not in LABELS for row in rows):
        raise ValueError(f"unknown {expected_split} target")
    class_counts = Counter(row["target"] for row in rows)
    if set(class_counts) != set(LABELS) or len(set(class_counts.values())) != 1:
        raise ValueError(f"{expected_split} examples are not balanced across the fixed taxonomy")
    for row in rows:
        media = (sample_dir / row["media_path"]).resolve()
        if not media.is_relative_to(sample_dir.resolve()) or not media.is_file():
            raise ValueError(f"missing or escaped media file: {media}")
        if sha256_file(media) != row["media_sha256"]:
            raise ValueError(f"media hash mismatch: {media}")
    return rows


def _check_groups(train: list[dict], validation: list[dict]) -> None:
    if {row["speaker_group_sha256"] for row in train} & {
        row["speaker_group_sha256"] for row in validation
    }:
        raise ValueError("train and validation speakers overlap")
    if {row["media_sha256"] for row in train} & {row["media_sha256"] for row in validation}:
        raise ValueError("train and validation audio content overlaps")


def run(feature_cache: Path, sample_dir: Path, output_dir: Path) -> dict[str, Any]:
    feature_cache, sample_dir, output_dir = (
        feature_cache.resolve(), sample_dir.resolve(), output_dir.resolve()
    )
    train_rows = _load_rows(sample_dir / "train.jsonl", "train", sample_dir)
    val_rows = _load_rows(sample_dir / "validation.jsonl", "validation", sample_dir)
    _check_groups(train_rows, val_rows)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=True)
    if cache.get("source_revision") != SOURCE or cache.get("encoder_revision") != AUDIO_REVISION:
        raise ValueError("frozen feature cache source or encoder revision mismatch")
    if cache["train_ids"] != [str(row["id"]) for row in train_rows] or cache[
        "validation_ids"
    ] != [str(row["id"]) for row in val_rows]:
        raise ValueError("feature cache IDs/order do not match the frozen sample manifests")
    train_targets = torch.tensor([LABELS.index(row["target"]) for row in train_rows])
    val_targets = torch.tensor([LABELS.index(row["target"]) for row in val_rows])
    if (
        cache["train_targets"] != train_targets.tolist()
        or cache["validation_targets"] != val_targets.tolist()
    ):
        raise ValueError("feature cache targets do not match sample manifests")
    x_train = cache["train_features"].float()
    x_val = cache["validation_features"].float()
    if (
        x_train.shape != (len(train_rows), 384)
        or x_val.shape != (len(val_rows), 384)
        or not torch.isfinite(x_train).all()
        or not torch.isfinite(x_val).all()
    ):
        raise ValueError("invalid frozen Whisper feature matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")

    seed = 17
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    model = TextFreeAudioReadout(384, LABELS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    history = []
    best_nll = math.inf
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, 9):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), 64):
            indexes = torch.tensor(order[start : start + 64])
            loss = F.cross_entropy(model(x_train[indexes]), train_targets[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val)
        metrics = _metrics(val_logits, val_targets)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **metrics})
        if metrics["nll"] < best_nll:
            best_nll = float(metrics["nll"])
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_logits = model(x_train)
        val_logits = model(x_val)
    readout_path = output_dir / "best-readout.pt"
    torch.save(best_state, readout_path)
    reloaded = TextFreeAudioReadout(384, LABELS).eval()
    reloaded.load_state_dict(torch.load(readout_path, map_location="cpu", weights_only=True))
    reloaded_logits = reloaded(x_val)
    if not torch.equal(reloaded_logits, val_logits):
        raise ValueError("save/reload changed text-free audio logits")
    predictions_path = output_dir / "validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (row, target, logits) in enumerate(
            zip(val_rows, val_targets.tolist(), val_logits, strict=True)
        ):
            probabilities = torch.softmax(logits.float(), dim=-1).tolist()
            options = [
                {"option_id": f"speech-command:{label}", "class_id": label} for label in LABELS
            ]
            decision = model.decide(x_val[index], options)
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "speaker_group_sha256": row["speaker_group_sha256"],
                        "target_class_id": LABELS[target],
                        "predicted_option_id": decision["selected_option_id"],
                        "decision": decision,
                        "probabilities": probabilities,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    params = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "seed": seed,
        "epochs": 8,
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint_selection": "minimum validation NLL",
        "feature_width": 384,
        "classes": LABELS,
    }
    per_class = {}
    for class_index, class_id in enumerate(LABELS):
        mask = val_targets == class_index
        per_class[class_id] = _metrics(val_logits[mask], val_targets[mask])
    report = {
        "schema_version": 1,
        "status": "complete_text_free_audio_development_probe",
        "task_claim": (
            "Closed-set 10-way English Speech Commands recognition from frozen audio features "
            "only; not general speech comprehension or product readiness."
        ),
        "seed": seed,
        "training": config,
        "training_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_revision": SOURCE,
        "audio_encoder_revision": AUDIO_REVISION,
        "input_contract": (
            "single 384-wide frozen Whisper encoder mean-pooled feature; no question text, "
            "candidate text or text encoder"
        ),
        "class_ids": LABELS,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "train_unique_speakers": len({row["speaker_group_sha256"] for row in train_rows}),
        "validation_unique_speakers": len({row["speaker_group_sha256"] for row in val_rows}),
        "train_validation_speaker_overlap": 0,
        "train_validation_media_hash_overlap": 0,
        "train_jsonl_sha256": sha256_file(sample_dir / "train.jsonl"),
        "validation_jsonl_sha256": sha256_file(sample_dir / "validation.jsonl"),
        "feature_cache_sha256": sha256_file(feature_cache),
        "train_ids_sha256": hashlib.sha256("\n".join(cache["train_ids"]).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(cache["validation_ids"]).encode()
        ).hexdigest(),
        "trainable_parameters": params,
        "optimizer_updates": sum(
            math.ceil(len(train_rows) / config["batch_size"]) for _ in range(8)
        ),
        "history": history,
        "selected_epoch": best_epoch,
        "selected_train_metrics": _metrics(train_logits, train_targets),
        "selected_validation_metrics": _metrics(val_logits, val_targets),
        "selected_validation_metrics_by_class": per_class,
        "save_reload_exact_logits": True,
        "training_seconds_cpu": training_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_used": False,
        "sealed_audit_loaded": False,
        "validation_limit": (
            "This exact validation split was previously used to select the audio+text readout. "
            "Results are a matched development comparison, not blind evidence."
        ),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "readout_implementation_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "src"
            / "tiny_omni_decision"
            / "textfree_audio.py"
        ),
        "readout_sha256": sha256_file(readout_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.feature_cache, args.sample_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    _main()
