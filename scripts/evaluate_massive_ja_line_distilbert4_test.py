"""One-time official-test evaluation of the frozen 4-layer Japanese encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

DATA_REVISION = "5092332a2835c4d29c617ef6d118123a1abecd19"
TRAIN_SHA256 = "ab55ddd41e4cf0f07f65bddc84432bea40ceb36caaa94c6edf15499ddf6a1270"
SPLIT_MANIFEST_SHA256 = "d9fdcc0235b3caecac462574e6111a6c1bf59102eab3ce39361c7fb42af9ed1d"
TEST_BYTES = 220_619
TEST_SHA256 = "985d95a0ec20724bc8f07e17e375f73ced6572f9670a3a525a19065736cde012"
MODEL_REVISION = "c32d05753d0d4be0f60482291a9452ec7e6f7d68"
MODEL_SHA256 = "9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb"
EXPECTED_ENCODER_PARAMETERS = 53_912_064
MAX_LENGTH = 128
EPOCHS = 8
SEED = 17


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_texts(values: list[str]) -> str:
    payload = "".join(value + "\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_directory(path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    entries = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        entries.append({
            "path": relative,
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, sum(item["bytes"] for item in entries), entries


def process_peak_memory_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().peak_wset)


def normalize_utterance(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("utterance must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class IntentReadout(nn.Module):
    def __init__(self, hidden_size: int, class_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, class_count),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def check_model(model_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    weight_path = model_dir / "pytorch_model.bin"
    if sha256_file(weight_path) != MODEL_SHA256:
        raise ValueError("LINE DistilBERT checkpoint hash mismatch")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "distilbert" or int(config.n_layers) != 6:
        raise ValueError("expected the pinned six-layer DistilBERT source config")
    config.n_layers = 4
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True,
    )
    model, loading_info = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        use_safetensors=False,
        output_loading_info=True,
    )
    model.requires_grad_(False)
    model.eval()
    missing = sorted(str(key) for key in loading_info["missing_keys"])
    unexpected = sorted(str(key) for key in loading_info["unexpected_keys"])
    if missing:
        raise ValueError(f"checkpoint failed to initialize encoder keys: {missing}")
    if len(unexpected) != 38:
        raise ValueError(f"expected 38 discarded upper-layer/MLM keys, got {len(unexpected)}")
    for key in unexpected:
        allowed = key.startswith("vocab_") or any(
            f"distilbert.transformer.layer.{index}." in key for index in (4, 5)
        )
        if not allowed:
            raise ValueError(f"unexpected checkpoint key outside truncation/MLM head: {key}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_ENCODER_PARAMETERS:
        raise ValueError(f"unexpected four-layer encoder parameter count: {parameters}")
    if any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise ValueError("encoder is not completely frozen")
    info = {
        "revision": MODEL_REVISION,
        "source_weight_sha256": MODEL_SHA256,
        "source_weight_bytes": weight_path.stat().st_size,
        "source_layers": 6,
        "effective_layers": 4,
        "parameters": parameters,
        "trainable_parameters": 0,
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_key_count": len(unexpected),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "embedding_rows": int(model.get_input_embeddings().num_embeddings),
    }
    return tokenizer, model, info


def encode_texts(
    texts: list[str], tokenizer: Any, model: nn.Module, batch_size: int
) -> tuple[Tensor, float, dict[str, int]]:
    features: list[Tensor] = []
    truncated = 0
    maximum_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            raw = tokenizer(batch_texts, truncation=False, padding=False)
            lengths = [len(ids) for ids in raw["input_ids"]]
            maximum_tokens = max(maximum_tokens, max(lengths, default=0))
            truncated += sum(length > MAX_LENGTH for length in lengths)
            batch = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            if int(batch["input_ids"].max()) >= model.get_input_embeddings().num_embeddings:
                raise ValueError("tokenizer emitted an ID outside the model embedding table")
            output = model(**batch).last_hidden_state[:, 0].float().cpu()
            if not torch.isfinite(output).all():
                raise ValueError("encoder emitted non-finite CLS features")
            features.append(output)
    elapsed = time.perf_counter() - started
    return torch.cat(features), elapsed, {
        "maximum_untruncated_tokens": maximum_tokens,
        "truncated_rows": truncated,
    }


def metrics(logits: Tensor, targets: Tensor, intent_names: list[str]) -> dict[str, Any]:
    probabilities = logits.float().softmax(-1)
    prediction = probabilities.argmax(-1)
    correct = prediction.eq(targets)
    confidence = probabilities.max(-1).values
    one_hot = F.one_hot(targets, len(intent_names)).float()
    ece = torch.zeros((), dtype=torch.float32)
    for index in range(15):
        low, high = index / 15, (index + 1) / 15
        mask = (confidence >= low) & (
            confidence < high if index < 14 else confidence <= high
        )
        if mask.any():
            ece += mask.float().mean() * (
                correct[mask].float().mean() - confidence[mask].mean()
            ).abs()
    per_intent: dict[str, Any] = {}
    for index, name in enumerate(intent_names):
        mask = targets == index
        per_intent[name] = {
            "count": int(mask.sum()),
            "accuracy": float(correct[mask].float().mean()) if mask.any() else None,
        }
    present = [item["accuracy"] for item in per_intent.values() if item["accuracy"] is not None]
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "macro_accuracy_present_intents": sum(present) / len(present),
        "intents_present": len(present),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "multiclass_brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece_15_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
        "per_intent": per_intent,
    }


def group_bootstrap_accuracy(
    correct: Tensor, group_ids: list[str], *, seed: int = 17, repeats: int = 2000
) -> dict[str, Any]:
    rows_by_group: dict[str, list[bool]] = defaultdict(list)
    for group, value in zip(group_ids, correct.tolist(), strict=True):
        rows_by_group[group].append(bool(value))
    groups = sorted(rows_by_group)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sampled = [rng.choice(groups) for _ in groups]
        correct_count = sum(sum(rows_by_group[group]) for group in sampled)
        row_count = sum(len(rows_by_group[group]) for group in sampled)
        values.append(correct_count / row_count)
    values.sort()
    return {
        "unit": "normalized-utterance content group",
        "groups": len(groups),
        "replicates": repeats,
        "seed": seed,
        "accuracy_percentile_95_interval": [
            values[int(0.025 * repeats)],
            values[int(0.975 * repeats)],
        ],
    }


def run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must remain hidden; this protocol is CPU-only")
    torch.set_num_threads(4)
    if sha256_file(args.train_jsonl) != TRAIN_SHA256:
        raise ValueError("frozen training JSONL hash mismatch")
    if sha256_file(args.split_manifest) != SPLIT_MANIFEST_SHA256:
        raise ValueError("frozen split manifest hash mismatch")
    if sha256_file(args.test_parquet) != TEST_SHA256:
        raise ValueError("pinned official test Parquet hash mismatch")
    if args.test_parquet.stat().st_size != TEST_BYTES:
        raise ValueError("pinned official test Parquet byte length mismatch")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("publisher_test_downloaded_or_opened") is not False:
        raise ValueError("source manifest does not prove test was previously unopened")
    intent_names = manifest["intent_names"]
    if len(intent_names) != 60 or len(set(intent_names)) != 60:
        raise ValueError("pinned MASSIVE taxonomy must contain 60 unique intents")
    train_rows = read_jsonl(args.train_jsonl)
    train_ids = {str(row["source_id"]) for row in train_rows}
    train_texts = [normalize_utterance(str(row["utterance"])) for row in train_rows]
    train_text_set = set(train_texts)
    if len(train_rows) != 11_228 or len(train_ids) != len(train_rows):
        raise ValueError("frozen train split row count or source IDs changed")
    for row in train_rows:
        target = int(row["target_index"])
        if not 0 <= target < len(intent_names) or intent_names[target] != row["intent"]:
            raise ValueError("train target index/name mismatch")

    test_schema = pq.read_schema(args.test_parquet)
    required = {"id", "utt", "intent"}
    if not required.issubset(set(test_schema.names)):
        missing_fields = required - set(test_schema.names)
        raise ValueError(f"official test schema lacks required fields: {missing_fields}")
    # Preflight reads inputs only; the intent column is not touched before fit.
    input_table = pq.read_table(args.test_parquet, columns=["id", "utt"])
    test_ids = [str(value) for value in input_table["id"].to_pylist()]
    test_texts_raw = [str(value) for value in input_table["utt"].to_pylist()]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("duplicate source IDs within official test")
    normalized_test = [normalize_utterance(text) for text in test_texts_raw]
    keep_indices = [
        index
        for index, (source_id, normalized) in enumerate(zip(test_ids, normalized_test, strict=True))
        if source_id not in train_ids and normalized not in train_text_set
    ]
    excluded_id = sum(test_ids[index] in train_ids for index in range(len(test_ids)))
    excluded_text = sum(
        normalized_test[index] in train_text_set and test_ids[index] not in train_ids
        for index in range(len(test_ids))
    )
    kept_ids = [test_ids[index] for index in keep_indices]
    kept_texts = [test_texts_raw[index] for index in keep_indices]
    kept_normalized = [normalized_test[index] for index in keep_indices]
    content_group_ids = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in kept_normalized
    ]
    if not keep_indices:
        raise ValueError("no non-overlapping official test rows remain")
    preflight = {
        "source_revision": DATA_REVISION,
        "test_file_bytes": args.test_parquet.stat().st_size,
        "test_file_sha256": sha256_file(args.test_parquet),
        "test_rows_total": len(test_ids),
        "test_rows_excluded_source_id_overlap": excluded_id,
        "test_rows_excluded_normalized_utterance_overlap": excluded_text,
        "test_rows_scored": len(keep_indices),
        "test_unique_normalized_content_groups_scored": len(set(content_group_ids)),
        "ordered_test_id_hash": sha256_texts(kept_ids),
        "ordered_test_normalized_content_hash": sha256_texts(kept_normalized),
        "intent_labels_read": False,
    }
    write_json(output / "test-preflight-inputs-only.json", preflight)

    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer, encoder, model_info = check_model(args.model_dir)
    feature_start = time.perf_counter()
    train_features, train_feature_seconds, train_token_info = encode_texts(
        [str(row["utterance"]) for row in train_rows], tokenizer, encoder, 64
    )
    training_feature_seconds = time.perf_counter() - feature_start
    train_targets = torch.tensor([int(row["target_index"]) for row in train_rows], dtype=torch.long)

    readout = IntentReadout(int(encoder.config.hidden_size), len(intent_names))
    trainable_parameters = sum(parameter.numel() for parameter in readout.parameters())
    if trainable_parameters != 106_172:
        raise ValueError(f"unexpected readout parameter count: {trainable_parameters}")
    optimizer = torch.optim.AdamW(readout.parameters(), lr=2e-3, weight_decay=1e-4)
    order_rng = random.Random(SEED)
    history = []
    fit_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        readout.train()
        losses = []
        for start in range(0, len(order), 64):
            indices = torch.tensor(order[start : start + 64], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(readout(train_features[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), len(indices)))
        examples_seen = sum(count for _, count in losses)
        history.append({
            "epoch": epoch,
            "train_ce": sum(loss_value * count for loss_value, count in losses)
            / examples_seen,
            "examples": examples_seen,
        })
    fit_seconds = time.perf_counter() - fit_start
    checkpoint = output / "final-epoch-8-readout.pt"
    torch.save(readout.state_dict(), checkpoint)

    # Reload the frozen, final-epoch readout before touching test labels.
    reloaded = IntentReadout(int(encoder.config.hidden_size), len(intent_names))
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        train_reload_logits = reloaded(train_features[: min(32, len(train_features))])
    if not torch.isfinite(train_reload_logits).all():
        raise ValueError("reloaded readout emitted non-finite output")

    # This is the first operation that reads test labels or computes test logits.
    test_table = pq.read_table(args.test_parquet, columns=["id", "utt", "intent"])
    full_ids = [str(value) for value in test_table["id"].to_pylist()]
    full_texts = [str(value) for value in test_table["utt"].to_pylist()]
    label_values = test_table["intent"].to_pylist()
    if full_ids != test_ids or full_texts != test_texts_raw:
        raise ValueError("test inputs changed between preflight and scoring")
    name_to_index = {name: index for index, name in enumerate(intent_names)}
    targets: list[int] = []
    for index in keep_indices:
        label = label_values[index]
        target = name_to_index[label] if isinstance(label, str) else int(label)
        if not 0 <= target < len(intent_names):
            raise ValueError(f"test target out of range: {target}")
        targets.append(target)

    test_features, test_feature_seconds, test_token_info = encode_texts(
        kept_texts, tokenizer, encoder, 64
    )
    evaluation_start = time.perf_counter()
    with torch.inference_mode():
        test_logits = reloaded(test_features)
    if not torch.isfinite(test_logits).all():
        raise ValueError("reloaded model emitted non-finite test logits")
    evaluation_seconds = time.perf_counter() - evaluation_start
    target_tensor = torch.tensor(targets, dtype=torch.long)
    test_metrics = metrics(test_logits, target_tensor, intent_names)
    correct = test_logits.argmax(-1).eq(target_tensor)
    group_ci = group_bootstrap_accuracy(correct, content_group_ids)
    probabilities = test_logits.float().softmax(-1)
    predictions = []
    for row_index, probability in enumerate(probabilities.tolist()):
        target = targets[row_index]
        prediction = max(range(len(intent_names)), key=probability.__getitem__)
        predictions.append({
            "sample_id": kept_ids[row_index],
            "normalized_content_group_id": content_group_ids[row_index],
            "target_index": target,
            "target_intent": intent_names[target],
            "prediction_index": prediction,
            "prediction_intent": intent_names[prediction],
            "probabilities": probability,
        })
    predictions_path = output / "test-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    train_features_path = output / "train-features.pt"
    torch.save(
        {"train_features": train_features, "train_targets": train_targets},
        train_features_path,
    )
    history_path = output / "training-history.json"
    write_json(history_path, history)
    model_directory_hash, model_directory_bytes, model_directory_files = hash_directory(
        args.model_dir
    )
    artifact_files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "run-report.json"
    )
    final_report = {
        "status": "complete_single_official_test_evaluation",
        "started_at_utc": args.started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "dataset": "AmazonScience/massive",
            "revision": DATA_REVISION,
            "license": "CC BY 4.0",
            "train_jsonl_sha256": sha256_file(args.train_jsonl),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "test_parquet_sha256": sha256_file(args.test_parquet),
            "test_preflight": preflight,
        },
        "model": model_info,
        "pretrained_model_directory": {
            "directory_sha256": model_directory_hash,
            "total_bytes": model_directory_bytes,
            "files": model_directory_files,
        },
        "task": "Japanese 60-intent classification; fixed probability vector in taxonomy order",
        "readout": {
            "architecture": "[CLS] 768 -> 128 tanh -> dropout(0.1) -> 60 logits",
            "trainable_parameters": trainable_parameters,
            "encoder_frozen": True,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "epochs": EPOCHS,
            "selected_epoch": 8,
            "selection": "fixed final epoch; no validation or test checkpoint selection",
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "batch_size": 64,
            "calibration": "none",
            "training_history": history,
        },
        "metrics": test_metrics,
        "group_bootstrap_accuracy": group_ci,
        "timing_seconds": {
            "training_feature_extraction": training_feature_seconds,
            "training_encoder_forward_only": train_feature_seconds,
            "readout_fit": fit_seconds,
            "test_feature_extraction": test_feature_seconds,
            "test_readout_scoring": evaluation_seconds,
        },
        "tokenization": {
            "training": train_token_info,
            "test": test_token_info,
            "tokenizer_vocab_size": model_info["tokenizer_vocab_size"],
            "embedding_rows": model_info["embedding_rows"],
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "transformers": metadata.version("transformers"),
            "pyarrow": metadata.version("pyarrow"),
            "fugashi": metadata.version("fugashi"),
            "sentencepiece": metadata.version("sentencepiece"),
            "unidic_lite": metadata.version("unidic-lite"),
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "peak_process_memory_bytes": process_peak_memory_bytes(),
        },
        "package": {
            "output_files_total_bytes": sum(path.stat().st_size for path in artifact_files),
            "output_files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_files
            ],
        },
        "artifact_sha256": {
            "script": sha256_file(Path(__file__)),
            "checkpoint": sha256_file(checkpoint),
            "predictions": sha256_file(predictions_path),
            "training_history": sha256_file(history_path),
            "train_features": sha256_file(train_features_path),
        },
        "test_policy": (
            "One-time evaluation. Do not tune, select, calibrate, or repeat model "
            "evaluation using this publisher test."
        ),
        "project_sealed_audit_opened": False,
    }
    write_json(output / "run-report.json", final_report)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    args.started_at_utc = datetime.now(UTC).isoformat()
    try:
        result = run(args, output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        write_json(output / "attempt-status.json", {
            "status": "incomplete",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "official_test_labels_scored": False,
        })
        raise


if __name__ == "__main__":
    main()
