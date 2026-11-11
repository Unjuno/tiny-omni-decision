"""Train only a small reader on frozen Perception Test train features.

This script intentionally accepts no public-validation path. Its internal
selection set is formed only from source-train video groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from scripts.extract_perception_test_text_features import sha256_file
from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout

SEED = 17
SELECTION_FRACTION = 0.15
BATCH_QUESTIONS = 16
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VIDEO_TOKEN_SHAPE = (8, 14, 14, 384)
TEXT_FEATURE_SHAPE = (3, 384)
TRAIN_JSONL_SHA256 = "2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7"
TEXT_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
VIDEO_MODEL_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
PREFLIGHT_SHA256 = "e04504116449b100d9c415d63b5fa0204c550c83cf75ef2b4d6fc523c476c910"


def split_training_groups(
    rows: list[dict[str, Any]], *, seed: int = SEED, selection_fraction: float = SELECTION_FRACTION
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not rows or not 0 < selection_fraction < 1:
        raise ValueError("nonempty training rows and a fraction in (0,1) are required")
    groups = [str(row.get("source_video_id", "")) for row in rows]
    if any(not group for group in groups):
        raise ValueError("every training row needs a source video ID")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("at least two video groups are required")
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(np.asarray(unique_groups, dtype=str)).tolist()
    selection_count = min(
        len(unique_groups) - 1, math.ceil(selection_fraction * len(unique_groups))
    )
    selection_groups = set(permuted[:selection_count])
    train_groups = set(unique_groups) - selection_groups
    train_indices = [index for index, group in enumerate(groups) if group in train_groups]
    selection_indices = [index for index, group in enumerate(groups) if group in selection_groups]
    if not train_indices or not selection_indices:
        raise ValueError("group split produced an empty partition")
    if {groups[index] for index in train_indices} & {groups[index] for index in selection_indices}:
        raise AssertionError("video groups crossed the internal split")
    return (
        train_indices,
        selection_indices,
        {
            "seed": seed,
            "fraction": selection_fraction,
            "selection_group_count": len(selection_groups),
            "train_group_count": len(train_groups),
            "selection_group_ids_sha256": hashlib.sha256(
                "\n".join(sorted(selection_groups)).encode("utf-8")
            ).hexdigest(),
            "train_group_ids_sha256": hashlib.sha256(
                "\n".join(sorted(train_groups)).encode("utf-8")
            ).hexdigest(),
            "group_key": "source_video_id",
        },
    )


def selection_improved(current_nll, current_brier, best_nll, best_brier) -> bool:
    return current_nll < best_nll - MIN_DELTA or (
        abs(current_nll - best_nll) <= MIN_DELTA and current_brier < best_brier
    )


def should_stop(epochs_without_improvement: int) -> bool:
    if epochs_without_improvement < 0:
        raise ValueError("non-improvement count cannot be negative")
    return epochs_without_improvement >= EARLY_STOPPING_PATIENCE


def _rows(corpus_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    path = corpus_dir / "train.jsonl"
    if (
        sha256_file(path) != TRAIN_JSONL_SHA256
        or manifest["train"]["records_sha256"] != TRAIN_JSONL_SHA256
    ):
        raise ValueError("train corpus hash differs from the fixed protocol")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 7392 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unexpected train row count or duplicate sample IDs")
    for row in rows:
        target = int(row["target_option_index"])
        if (
            row.get("split") != "train"
            or len(row.get("options", [])) != 3
            or target not in (0, 1, 2)
        ):
            raise ValueError(f"invalid train Decision row: {row.get('id')}")
        if row["target"] != row["options"][target]:
            raise ValueError(f"target option index mismatch: {row['id']}")
    return rows, manifest


def _load_text_features(path: Path, report_path: Path, rows: list[dict[str, Any]]):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETE"
        or report.get("split") != "train"
        or report.get("records_sha256") != TRAIN_JSONL_SHA256
        or report.get("model_files_sha256", {}).get("model.safetensors") != TEXT_MODEL_SHA256
        or report.get("labels_used") is not False
        or report.get("official_test_opened") is not False
        or sha256_file(path) != report.get("feature_sha256")
    ):
        raise ValueError("text feature report/hash does not match fixed train contract")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        video_ids = archive["video_ids"].astype(str)
        features = archive["features"]
    if (
        ids.tolist() != [row["id"] for row in rows]
        or video_ids.tolist() != [row["source_video_id"] for row in rows]
        or features.shape != (len(rows), *TEXT_FEATURE_SHAPE)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("text features do not align exactly with frozen train rows")
    return features, report


class VideoFeatureCache:
    def __init__(
        self,
        directory: Path,
        manifest_path: Path,
        max_entries: int = 32,
        expected_split: str = "train",
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("split") != expected_split
            or manifest.get("preflight_sha256") != PREFLIGHT_SHA256
            or manifest.get("checkpoint_sha256") != VIDEO_MODEL_SHA256
            or tuple(manifest.get("token_shape_per_video", [])) != VIDEO_TOKEN_SHAPE
            or len(manifest.get("media_receipt_set_sha256", "")) != 64
        ):
            raise ValueError("VideoMAE feature manifest does not match the fixed train contract")
        self.directory = directory
        self.max_entries = max_entries
        self.receipts = {row["video_id"]: row for row in manifest.get("feature_receipts", [])}
        if len(self.receipts) != len(manifest.get("feature_receipts", [])) or not self.receipts:
            raise ValueError("VideoMAE receipts must be nonempty and unique")
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        for video_id, receipt in self.receipts.items():
            path = directory / f"{video_id}.npy"
            if not path.is_file() or sha256_file(path) != receipt.get("feature_sha256"):
                raise ValueError(f"VideoMAE feature shard hash mismatch: {video_id}")

    def get(self, video_id: str) -> np.ndarray:
        if video_id not in self.receipts:
            raise ValueError(f"missing frozen video feature: {video_id}")
        if video_id not in self.cache:
            array = np.load(self.directory / f"{video_id}.npy", allow_pickle=False)
            if (
                array.shape != VIDEO_TOKEN_SHAPE
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid frozen video tokens: {video_id}")
            self.cache[video_id] = array.reshape(-1, VIDEO_TOKEN_SHAPE[-1])
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)
        value = self.cache.pop(video_id)
        self.cache[video_id] = value
        return value


def _batch(indices, rows, text_features, video_store, device):
    video = np.stack([video_store.get(rows[index]["source_video_id"]) for index in indices])
    queries = text_features[indices]
    targets = np.asarray([int(rows[index]["target_option_index"]) for index in indices])
    return (
        torch.from_numpy(video.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(queries.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(targets).to(device=device, dtype=torch.long),
    )


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    nll = -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (confidence < upper if index < 14 else confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": float(ece),
    }


def _predict(model, indices, rows, text_features, video_store, device):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_QUESTIONS):
            batch_indices = indices[start : start + BATCH_QUESTIONS]
            video, queries, _ = _batch(batch_indices, rows, text_features, video_store, device)
            chunks.append(model(video, queries).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _metrics_by_type(logits, indices, rows):
    targets = np.asarray([int(rows[index]["target_option_index"]) for index in indices])
    output = {"overall": _metrics(logits, targets)}
    kinds = [rows[index]["reasoning_type"] for index in indices]
    for kind in sorted(set(kinds)):
        selected = np.asarray([value == kind for value in kinds])
        output[kind] = _metrics(logits[selected], targets[selected])
    return output


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def train(
    *,
    corpus_dir: Path,
    text_features: Path,
    text_report: Path,
    video_feature_dir: Path,
    video_feature_manifest: Path,
    output_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    for path in (corpus_dir, text_features, text_report, video_feature_dir, video_feature_manifest):
        path.resolve(strict=True)
    if output_dir.drive.upper() != "D:" or output_dir.exists():
        raise ValueError("use a new, nonexisting output directory on D:")
    if device_name not in {"cuda", "cpu"}:
        raise ValueError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows, corpus_manifest = _rows(corpus_dir)
    text, text_meta = _load_text_features(text_features, text_report, rows)
    video_store = VideoFeatureCache(video_feature_dir, video_feature_manifest)
    all_groups = {row["source_video_id"] for row in rows}
    if set(video_store.receipts) != all_groups:
        raise ValueError("train video feature IDs differ from the frozen train corpus")
    train_indices, selection_indices, split_meta = split_training_groups(rows)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    model = PerceptionVideoDecisionReadout().to(device).train()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    config = {
        "schema_version": 1,
        "experiment_id": "r2-perception-test-mcqa-path-a-20261111",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "reader_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "src"
            / "tiny_omni_decision"
            / "perception_video_readout.py"
        ),
        "text_feature_script_sha256": sha256_file(
            Path(__file__).resolve().with_name("extract_perception_test_text_features.py")
        ),
        "command_line": sys.argv,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "corpus_dir": str(corpus_dir.resolve()),
        "text_feature_file": str(text_features.resolve()),
        "video_feature_manifest": str(video_feature_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": SEED,
        "corpus_train_sha256": TRAIN_JSONL_SHA256,
        "text_feature_sha256": text_meta["feature_sha256"],
        "video_feature_manifest_sha256": sha256_file(video_feature_manifest),
        "video_media_receipt_set_sha256": json.loads(
            video_feature_manifest.read_text(encoding="utf-8")
        )["media_receipt_set_sha256"],
        "video_model_checkpoint_sha256": VIDEO_MODEL_SHA256,
        "text_model_checkpoint_sha256": TEXT_MODEL_SHA256,
        "internal_group_split": split_meta,
        "reader_hidden_size": 192,
        "reader_heads": 4,
        "reader_trainable_parameters": trainable_parameters,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "three-option cross entropy",
        "batch_questions": BATCH_QUESTIONS,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "selection_min_delta": MIN_DELTA,
        "selection_rule": "lowest internal-selection NLL; Brier tie-break within min_delta",
        "public_validation_opened": False,
        "sealed_audit_opened": False,
        "device_requested": device_name,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "effective-config.json").write_text(
        json.dumps({**config, "config_sha256": config_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_path = output_dir / "training-history.jsonl"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_state = None
    best_nll = math.inf
    best_brier = math.inf
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    generator = np.random.default_rng(SEED)
    final_epoch = 0
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
        for epoch in range(1, MAX_EPOCHS + 1):
            final_epoch = epoch
            epoch_started = time.perf_counter()
            permutation = generator.permutation(np.asarray(train_indices, dtype=np.int64)).tolist()
            losses = []
            gradient_norms = []
            model.train()
            for start in range(0, len(permutation), BATCH_QUESTIONS):
                indices = permutation[start : start + BATCH_QUESTIONS]
                video, queries, targets = _batch(indices, rows, text, video_store, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(video, queries)
                loss = F.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                grad_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().norm()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ]
                    )
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
                gradient_norms.append(float(grad_norm.item()))
                optimizer.step()
                losses.append(float(loss.detach().item()))

            train_logits = _predict(model, train_indices, rows, text, video_store, device)
            selection_logits = _predict(model, selection_indices, rows, text, video_store, device)
            train_metrics = _metrics_by_type(train_logits, train_indices, rows)
            selection_metrics = _metrics_by_type(selection_logits, selection_indices, rows)
            current_nll = selection_metrics["overall"]["nll"]
            current_brier = selection_metrics["overall"]["brier"]
            improved = selection_improved(current_nll, current_brier, best_nll, best_brier)
            if improved:
                best_nll = current_nll
                best_brier = current_brier
                best_epoch = epoch
                best_metrics = selection_metrics
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            record = {
                "epoch": epoch,
                "optimizer_updates": math.ceil(len(train_indices) / BATCH_QUESTIONS),
                "consumed_questions_cumulative": len(train_indices) * epoch,
                "unique_train_questions": len(train_indices),
                "unique_train_video_groups": split_meta["train_group_count"],
                "mean_batch_ce": float(np.mean(losses)),
                "train": train_metrics,
                "internal_selection": selection_metrics,
                "learning_rate": LEARNING_RATE,
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "max_gradient_norm": float(np.max(gradient_norms)),
                "best_epoch": best_epoch,
                "best_selection_nll": best_nll,
                "best_selection_brier": best_brier,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_epoch_seconds": time.perf_counter() - epoch_started,
            }
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        **record["internal_selection"]["overall"],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
            if should_stop(epochs_without_improvement):
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("no finite selection checkpoint was produced")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _atomic_torch_save(
        {
            "state_dict": best_state,
            "selected_epoch": best_epoch,
            "seed": SEED,
            "config_sha256": config_hash,
        },
        output_dir / "best-checkpoint.pt",
    )
    _atomic_torch_save(
        {
            "state_dict": final_state,
            "final_epoch": final_epoch,
            "seed": SEED,
            "config_sha256": config_hash,
        },
        output_dir / "final-checkpoint.pt",
    )
    model.load_state_dict(best_state, strict=True)
    reloaded_logits = _predict(model, selection_indices, rows, text, video_store, device)
    reloaded = _metrics_by_type(reloaded_logits, selection_indices, rows)["overall"]
    for metric in ("nll", "brier", "accuracy"):
        if abs(reloaded[metric] - best_metrics["overall"][metric]) > 1e-7:
            raise RuntimeError(f"best checkpoint reload differs for {metric}")

    report = {
        **config,
        "config_sha256": config_hash,
        "status": "TRAIN_COMPLETE",
        "selected_epoch": best_epoch,
        "final_epoch": final_epoch,
        "epochs_run": final_epoch,
        "train_question_count": len(train_indices),
        "train_unique_video_groups": split_meta["train_group_count"],
        "internal_selection_question_count": len(selection_indices),
        "internal_selection_unique_video_groups": split_meta["selection_group_count"],
        "consumed_questions_all_epochs": len(train_indices) * final_epoch,
        "consumed_unique_train_questions": len(train_indices),
        "consumed_unique_train_video_groups": split_meta["train_group_count"],
        "selected_internal_metrics": best_metrics,
        "best_checkpoint_sha256": sha256_file(output_dir / "best-checkpoint.pt"),
        "final_checkpoint_sha256": sha256_file(output_dir / "final-checkpoint.pt"),
        "training_history_sha256": sha256_file(history_path),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "best_checkpoint_bytes": (output_dir / "best-checkpoint.pt").stat().st_size,
        "final_checkpoint_bytes": (output_dir / "final-checkpoint.pt").stat().st_size,
        "video_encoder_parameters_frozen": 22_153_286,
        "text_encoder_parameters_frozen": 17_389_824,
        "reader_trainable_parameters": trainable_parameters,
        "resident_parameters_for_this_video_text_path": 39_987_783,
        "official_test_opened": False,
        "sealed_audit_opened": False,
    }
    (output_dir / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--video-feature-dir", type=Path, required=True)
    parser.add_argument("--video-feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = train(
        corpus_dir=args.corpus_dir,
        text_features=args.text_features,
        text_report=args.text_report,
        video_feature_dir=args.video_feature_dir,
        video_feature_manifest=args.video_feature_manifest,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_epoch": report["selected_epoch"],
                "final_epoch": report["final_epoch"],
                "selected_internal_metrics": report["selected_internal_metrics"]["overall"],
                "best_checkpoint_sha256": report["best_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
