"""Train a small candidate scorer over frozen Whisper Tiny audio features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import sha256_file

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
        ):
            raise ValueError(f"expected mono 16-bit 16kHz PCM WAV: {path}")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if samples.size == 0:
        raise ValueError(f"empty audio file: {path}")
    return samples.astype(np.float32) / 32768.0


def _validate_splits(
    sample_dir: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]
) -> None:
    expected_source = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        if not rows or any(
            row.get("split") != split or row.get("source") != expected_source for row in rows
        ):
            raise ValueError(f"invalid or empty {split} sample")
        if set(row["target"] for row in rows) != set(LABELS):
            raise ValueError(f"{split} sample does not include every pinned keyword")
        counts = {label: sum(row["target"] == label for row in rows) for label in LABELS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split} sample is not balanced: {counts}")
        for row in rows:
            path = (sample_dir / row["media_path"]).resolve()
            if not path.is_relative_to(sample_dir.resolve()) or not path.is_file():
                raise ValueError(f"missing or escaped audio file: {path}")
            if sha256_file(path) != row["media_sha256"]:
                raise ValueError(f"audio SHA-256 mismatch: {path}")
    train_speakers = {row["speaker_group_sha256"] for row in train_rows}
    val_speakers = {row["speaker_group_sha256"] for row in val_rows}
    if train_speakers & val_speakers:
        raise ValueError("train and validation speaker groups overlap")
    train_audio = {row["media_sha256"] for row in train_rows}
    val_audio = {row["media_sha256"] for row in val_rows}
    if train_audio & val_audio:
        raise ValueError("train and validation audio content overlaps")


def _load_encoder(args: argparse.Namespace, device: torch.device):
    expected_revision = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    if args.audio_revision != expected_revision:
        raise ValueError(f"unexpected Whisper revision: {args.audio_revision}")
    encoder_model = (
        WhisperModel.from_pretrained(
            args.audio_model, local_files_only=True, torch_dtype=torch.float32
        )
        .eval()
        .to(device)
    )
    encoder = encoder_model.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    processor = WhisperFeatureExtractor.from_pretrained(args.audio_model, local_files_only=True)
    return encoder, encoder_model.config, processor


def _extract_features(
    rows: list[dict[str, Any]],
    sample_dir: Path,
    encoder: nn.Module,
    processor: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, list[str], list[int], float]:
    features: list[Tensor] = []
    ids: list[str] = []
    targets: list[int] = []
    started = time.perf_counter()
    encoder.eval()
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        waveforms = [_read_wav(sample_dir / row["media_path"]) for row in current]
        encoded = processor(
            waveforms,
            sampling_rate=16000,
            padding="max_length",
            max_length=processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        input_features = encoded.input_features.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encoder(input_features).last_hidden_state
            pooled_rows = []
            for index, waveform in enumerate(waveforms):
                valid_tokens = min(hidden.shape[1], max(1, math.ceil(len(waveform) / 320)))
                pooled_rows.append(hidden[index, :valid_tokens].float().mean(dim=0))
            pooled = torch.stack(pooled_rows)
        if pooled.shape[-1] != 384 or not torch.isfinite(pooled).all():
            raise ValueError("frozen Whisper encoder produced invalid pooled features")
        features.append(pooled.cpu())
        ids.extend(str(row["id"]) for row in current)
        targets.extend(LABELS.index(str(row["target"])) for row in current)
    return torch.cat(features), ids, targets, time.perf_counter() - started


def _option_embeddings(model_path: Path, revision: str, device: torch.device) -> tuple[Tensor, int]:
    expected_revision = "4ca70771034acceecb2e72475f72050fcdde4ddc"
    if revision != expected_revision:
        raise ValueError(f"unexpected MiniLM revision: {revision}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prompts = [f"Spoken English keyword: {label}." for label in LABELS]
    encoded = tokenizer(
        prompts, max_length=32, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
    if not torch.isfinite(pooled).all():
        raise ValueError("candidate text embeddings contain non-finite values")
    size = int(model.config.hidden_size)
    del model
    torch.cuda.empty_cache()
    return pooled, size


def _question_features(audio_features: Tensor, option_embeddings: Tensor) -> list[Tensor]:
    if audio_features.shape[-1] != option_embeddings.shape[-1]:
        raise ValueError("audio and text candidate embeddings must have the same dimension")
    questions = []
    for audio in audio_features:
        expanded = audio.unsqueeze(0).expand(len(LABELS), -1)
        questions.append(
            torch.cat(
                [
                    expanded,
                    option_embeddings,
                    (expanded - option_embeddings).abs(),
                    expanded * option_embeddings,
                ],
                dim=-1,
            )
        )
    return questions


def _predict(scorer: nn.Module, features: list[Tensor], device: torch.device) -> list[Tensor]:
    scorer.eval()
    with torch.inference_mode():
        return list(scorer(torch.stack(features).to(device)).cpu().unbind(0))


def _option_order_stability(
    scorer: nn.Module,
    audio_features: Tensor,
    option_embeddings: Tensor,
    targets: list[int],
    device: torch.device,
) -> dict[str, Any]:
    base_logits = torch.stack(
        _predict(scorer, _question_features(audio_features, option_embeddings), device)
    )
    orders = {
        "reverse": list(reversed(range(len(LABELS)))),
        "rotate": list(range(3, len(LABELS))) + list(range(3)),
        "fixed_shuffle": [6, 0, 8, 2, 9, 1, 4, 7, 3, 5],
    }
    details = {}
    for name, order in orders.items():
        permuted_logits = torch.stack(
            _predict(scorer, _question_features(audio_features, option_embeddings[order]), device)
        )
        canonical_logits = torch.empty_like(permuted_logits)
        canonical_logits[:, order] = permuted_logits
        delta = float((canonical_logits - base_logits).abs().max())
        predicted = canonical_logits.argmax(dim=-1).tolist()
        details[name] = {
            "max_abs_logit_delta_after_option_id_alignment": delta,
            "accuracy_after_reordering": sum(
                prediction == target for prediction, target in zip(predicted, targets, strict=True)
            )
            / len(targets),
        }
        if delta > 1e-6:
            raise ValueError(f"candidate scores changed with option order: {name}, delta={delta}")
    return details


def _train_head(
    scorer: CandidateScorer,
    features: list[Tensor],
    targets: list[int],
    validation_features: list[Tensor],
    validation_targets: list[int],
    device: torch.device,
    seed: int,
    epochs: int,
    batch_questions: int,
) -> tuple[dict[str, Tensor], list[dict[str, Any]], int, float]:
    scorer.to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(features)))
        rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), batch_questions):
            indexes = order[start : start + batch_questions]
            batch = torch.stack([features[index] for index in indexes]).to(device)
            targets_tensor = torch.tensor([targets[index] for index in indexes], device=device)
            logits = scorer(batch.reshape(-1, batch.shape[-1])).reshape(len(indexes), len(LABELS))
            loss = F.cross_entropy(logits, targets_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _predict(scorer, validation_features, device)
        metrics = _metrics(val_logits, validation_targets)
        history.append(
            {"epoch": epoch, "train_ce": sum(losses) / len(losses), "validation": metrics}
        )
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no state")
    return best_state, history, best_epoch, elapsed


def main() -> None:
    args = _args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded frozen audio feature probe requires local CUDA")
    if args.epochs < 1 or args.feature_batch_size < 1 or args.batch_questions < 1:
        raise ValueError("epochs and batch sizes must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    sample_dir = args.sample_dir.resolve()
    train_rows = _rows(sample_dir / "train.jsonl")
    validation_rows = _rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train_rows, validation_rows)
    encoder, model_config, processor = _load_encoder(args, device)
    train_features, train_ids, train_targets, train_seconds = _extract_features(
        train_rows, sample_dir, encoder, processor, device, args.feature_batch_size
    )
    validation_features, validation_ids, validation_targets, validation_seconds = _extract_features(
        validation_rows, sample_dir, encoder, processor, device, args.feature_batch_size
    )
    feature_cache = output_dir / "audio-features.pt"
    torch.save(
        {
            "source_revision": "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da",
            "encoder_revision": args.audio_revision,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "train_features": train_features,
            "validation_features": validation_features,
            "train_targets": train_targets,
            "validation_targets": validation_targets,
        },
        feature_cache,
    )
    del encoder
    torch.cuda.empty_cache()

    option_embeddings, _ = _option_embeddings(args.text_model, args.text_revision, device)
    train_questions = _question_features(train_features, option_embeddings.cpu())
    validation_questions = _question_features(validation_features, option_embeddings.cpu())
    scorer = CandidateScorer(train_questions[0].shape[-1])
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    best_state, history, best_epoch, training_seconds = _train_head(
        scorer,
        train_questions,
        train_targets,
        validation_questions,
        validation_targets,
        device,
        args.seed,
        args.epochs,
        args.batch_questions,
    )
    scorer.load_state_dict(best_state)
    train_inference_started = time.perf_counter()
    train_logits = _predict(scorer, train_questions, device)
    torch.cuda.synchronize(device)
    train_inference_seconds = time.perf_counter() - train_inference_started
    validation_inference_started = time.perf_counter()
    validation_logits = _predict(scorer, validation_questions, device)
    torch.cuda.synchronize(device)
    validation_inference_seconds = time.perf_counter() - validation_inference_started
    train_option_order = _option_order_stability(
        scorer, train_features, option_embeddings.cpu(), train_targets, device
    )
    validation_option_order = _option_order_stability(
        scorer, validation_features, option_embeddings.cpu(), validation_targets, device
    )
    torch.save(best_state, output_dir / "best-readout.pt")
    validation_predictions = []
    for row, target, logits in zip(
        validation_rows, validation_targets, validation_logits, strict=True
    ):
        probabilities = torch.softmax(logits.float(), dim=-1)
        validation_predictions.append(
            {
                "id": row["id"],
                "speaker_group_sha256": row["speaker_group_sha256"],
                "target": LABELS[target],
                "prediction": LABELS[int(probabilities.argmax())],
                "options": LABELS,
                "probabilities": probabilities.tolist(),
            }
        )
    prediction_path = output_dir / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_predictions), encoding="utf-8"
    )
    train_metrics = _metrics(train_logits, train_targets)
    val_metrics = _metrics(validation_logits, validation_targets)
    val_by_keyword = {
        label: _metrics(
            [
                logit
                for logit, row in zip(validation_logits, validation_rows, strict=True)
                if row["target"] == label
            ],
            [
                target
                for target, row in zip(validation_targets, validation_rows, strict=True)
                if row["target"] == label
            ],
        )
        for label in LABELS
    }
    report = {
        "schema_version": 1,
        "status": "complete_audio_only_development_probe",
        "task_claim": (
            "Balanced 10-way English Speech Commands keyword recognition; not general speech "
            "comprehension, Japanese ASR, or a sealed evaluation."
        ),
        "seed": args.seed,
        "source_revision": "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da",
        "sample_fetch_report_sha256": sha256_file(sample_dir / "fetch-report.json"),
        "train_sample_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "train_unique_examples": len(train_ids),
        "validation_unique_examples": len(validation_ids),
        "train_unique_speakers": len({row["speaker_group_sha256"] for row in train_rows}),
        "validation_unique_speakers": len({row["speaker_group_sha256"] for row in validation_rows}),
        "train_validation_speaker_overlap": 0,
        "train_validation_audio_hash_overlap": 0,
        "classes": LABELS,
        "class_counts_train": {
            label: sum(row["target"] == label for row in train_rows) for label in LABELS
        },
        "class_counts_validation": {
            label: sum(row["target"] == label for row in validation_rows) for label in LABELS
        },
        "encoder_repo_id": "openai/whisper-tiny",
        "encoder_revision": args.audio_revision,
        "encoder_model_config": {
            "d_model": model_config.d_model,
            "encoder_layers": model_config.encoder_layers,
            "max_source_positions": model_config.max_source_positions,
        },
        "encoder_model_safetensors_sha256": sha256_file(args.audio_model / "model.safetensors"),
        "feature_pooling": (
            "mean over first ceil(num_samples/320) Whisper encoder tokens; CUDA BF16 autocast "
            "with frozen FP32 checkpoint weights"
        ),
        "text_candidate_repo_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "text_candidate_revision": args.text_revision,
        "readout_parameters": trainable_parameters,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "epochs": args.epochs,
        "history": history,
        "best_epoch": best_epoch,
        "train_metrics_at_selected_checkpoint": train_metrics,
        "validation_metrics_at_selected_checkpoint": val_metrics,
        "validation_metrics_by_keyword": val_by_keyword,
        "candidate_order_stability_train": train_option_order,
        "candidate_order_stability_validation": validation_option_order,
        "uniform_10_way_baseline": {
            "expected_accuracy": 0.1,
            "nll": math.log(10),
            "brier": 0.9,
            "ece_15_bins": 0.0,
        },
        "audio_feature_cache_bytes": feature_cache.stat().st_size,
        "audio_feature_cache_sha256": sha256_file(feature_cache),
        "readout_sha256": sha256_file(output_dir / "best-readout.pt"),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "feature_extraction_train_seconds": train_seconds,
        "feature_extraction_validation_seconds": validation_seconds,
        "readout_training_seconds": training_seconds,
        "selected_checkpoint_train_inference_seconds": train_inference_seconds,
        "selected_checkpoint_validation_inference_seconds": validation_inference_seconds,
        "selected_checkpoint_validation_ms_per_example": (
            1000 * validation_inference_seconds / len(validation_rows)
        ),
        "device": torch.cuda.get_device_name(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
        "note": (
            "One seed; speaker-grouped official validation is used only for early selection and "
            "result reporting. Do not infer product-level audio quality."
        ),
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
