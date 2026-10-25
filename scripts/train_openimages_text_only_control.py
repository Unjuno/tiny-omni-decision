"""Fit one fixed-budget text-only control for the Open Images candidate task."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
TEXT_SIZE = 384
HIDDEN_SIZE = 128
READOUT_PARAMETERS = 49_409


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _linear_scorer_parameters(input_size: int, hidden_size: int) -> int:
    return input_size * hidden_size + hidden_size + hidden_size + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_openimages_videoprism.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    features_dir = args.features_dir.resolve(strict=True)
    source_run = args.source_run_dir.resolve(strict=True)
    text_model = args.text_model.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("text-only control artifacts must be stored outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite text-only output: {output}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer

    from scripts.train_openimages_frozen_probe import (
        _quality,
        _text_features,
        _validate_rows,
    )
    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    config = json.loads((features_dir / "config.json").read_text(encoding="utf-8"))
    source_metadata_path = source_run / "run-metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("experiment_id") != "path-a-openimages-videoprism-minilm-seed17":
        raise ValueError("unexpected frozen multimodal source run")
    if source_metadata.get("sealed_audit_loaded") is not False:
        raise ValueError("source run does not certify sealed-audit isolation")
    if _sha256(config_path) != source_metadata["config_sha256"]:
        raise ValueError("task config differs from the source run")
    if _sha256(features_dir / "config.json") != source_metadata["feature_config_sha256"]:
        raise ValueError("feature config differs from the source run")
    for split in ("train", "validation"):
        if (
            _sha256(features_dir / f"{split}-rows.jsonl")
            != config["output_hashes"][f"{split}-rows.jsonl"]
        ):
            raise ValueError(f"{split} rows differ from the frozen feature receipt")
    if _sha256(text_model / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM weights differ from the pinned revision")
    if source_metadata["text_encoder"]["weights_sha256"] != TEXT_WEIGHTS_SHA256:
        raise ValueError("source run used a different MiniLM checkpoint")

    train_rows = _read_rows(features_dir / "train-rows.jsonl")
    val_rows = _read_rows(features_dir / "validation-rows.jsonl")
    _validate_rows(train_rows, val_rows)
    validation_ids_sha256 = hashlib.sha256(
        "\n".join(str(row["sample_id"]) for row in val_rows).encode()
    ).hexdigest()
    if validation_ids_sha256 != source_metadata["split_ids_sha256"]["validation"]:
        raise ValueError("validation row order differs from the original run")

    torch.set_num_threads(4)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    encoder = AutoModel.from_pretrained(text_model, local_files_only=True).eval()
    encoder.requires_grad_(False)
    text_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if text_parameter_count != 17_389_824:
        raise ValueError(f"unexpected MiniLM parameter count: {text_parameter_count}")

    started = time.perf_counter()
    train_text = _text_features(train_rows, tokenizer, encoder)
    val_text = _text_features(val_rows, tokenizer, encoder)
    feature_seconds = time.perf_counter() - started
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("MiniLM unexpectedly has trainable parameters")
    del encoder, tokenizer

    train_x = torch.from_numpy(train_text)
    val_x = torch.from_numpy(val_text)
    train_y = torch.tensor(
        [row["options"].index(row["target"]) for row in train_rows], dtype=torch.long
    )
    val_y = torch.tensor(
        [row["options"].index(row["target"]) for row in val_rows], dtype=torch.long
    )
    scorer = FrozenFeatureCandidateScorer(TEXT_SIZE, hidden_size=HIDDEN_SIZE).cpu()
    readout_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    if readout_parameters != READOUT_PARAMETERS or readout_parameters != _linear_scorer_parameters(
        TEXT_SIZE, HIDDEN_SIZE
    ):
        raise ValueError(f"unexpected text-only readout parameter count: {readout_parameters}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(17)
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, 9):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            logits = scorer(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            train_logits = scorer(train_x)
            val_logits = scorer(val_x)
        epoch_result = {
            "epoch": epoch,
            "train_cross_entropy": float(np.mean(losses)),
            "train_metrics": _quality(train_logits, train_y),
            "validation_metrics": _quality(val_logits, val_y),
        }
        history.append(epoch_result)
    training_seconds = time.perf_counter() - training_started

    # Fixed final epoch only: this control has no validation-based selector.
    final_state = {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}
    scorer.eval()
    with torch.inference_mode():
        final_val_logits = scorer(val_x).float()
        final_train_logits = scorer(train_x).float()
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "final-text-only-readout.pt"
    torch.save(final_state, state_path)
    scorer.load_state_dict(
        torch.load(state_path, map_location="cpu", weights_only=True), strict=True
    )
    with torch.inference_mode():
        reloaded_logits = scorer(val_x).float()
    replay_delta = float((reloaded_logits - final_val_logits).abs().max())
    predictions_path = output / "validation-predictions.jsonl"
    prediction_rows = []
    probabilities = torch.softmax(reloaded_logits, dim=-1)
    for index, row in enumerate(val_rows):
        values = probabilities[index].tolist()
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "target_label": row["target_label"],
                "target": row["target"],
                "options": row["options"],
                "target_position": int(val_y[index]),
                "prediction": row["options"][int(np.argmax(values))],
                "probabilities": values,
            }
        )
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    train_text_path = output / "train-text-features.npy"
    val_text_path = output / "validation-text-features.npy"
    np.save(train_text_path, train_text, allow_pickle=False)
    np.save(val_text_path, val_text, allow_pickle=False)

    metrics_by_target = {}
    for label in sorted({str(row["target_label"]) for row in val_rows}):
        indices = [index for index, row in enumerate(val_rows) if str(row["target_label"]) == label]
        metrics_by_target[label] = _quality(reloaded_logits[indices], val_y[indices])
    result = {
        "schema_version": 1,
        "experiment_id": "path-a-openimages-text-only-control-seed17",
        "source_run_dir": str(source_run),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": _sha256(Path(__file__)),
        "source_metadata_sha256": _sha256(source_metadata_path),
        "task_config_sha256": _sha256(config_path),
        "feature_config_sha256": _sha256(features_dir / "config.json"),
        "training_rows_sha256": _sha256(features_dir / "train-rows.jsonl"),
        "validation_rows_sha256": _sha256(features_dir / "validation-rows.jsonl"),
        "validation_feature_source_sha256": source_metadata["feature_hashes"]["validation"],
        "validation_sample_order_sha256": validation_ids_sha256,
        "text_encoder": {
            "repo_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "revision": TEXT_REVISION,
            "weights_sha256": TEXT_WEIGHTS_SHA256,
            "parameters": text_parameter_count,
            "frozen": True,
        },
        "image_input_used": False,
        "sealed_audit_loaded": False,
        "training_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "seed": 17,
        "device": "CPU",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_questions": 64,
        "epochs": 8,
        "checkpoint_selection": "none; fixed final epoch 8",
        "readout_parameters": readout_parameters,
        "total_resident_parameters": text_parameter_count + readout_parameters,
        "history": history,
        "final_train_metrics": _quality(final_train_logits, train_y),
        "final_validation_metrics": _quality(reloaded_logits, val_y),
        "final_validation_metrics_by_target_label": metrics_by_target,
        "final_epoch_validation_reload_max_logit_delta": replay_delta,
        "text_feature_extraction_seconds": feature_seconds,
        "readout_training_seconds": training_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            name: _sha256(output / name)
            for name in (
                "final-text-only-readout.pt",
                "validation-predictions.jsonl",
                "train-text-features.npy",
                "validation-text-features.npy",
            )
        },
        "interpretation": (
            "single task-shortcut diagnostic on an exposed validation split; "
            "not a blind text or image quality estimate"
        ),
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    (output / "run-metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
