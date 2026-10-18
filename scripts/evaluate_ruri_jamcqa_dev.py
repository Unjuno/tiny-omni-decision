"""Run a frozen zero-shot Japanese multiple-choice diagnostic on JamC-QA-V2 dev only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.dataset import sha256_file

EXPECTED_MODEL_REVISION = "24899e5de370b56d179604a007c0d727bf144504"
EXPECTED_MODEL_SHA256 = "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832"
EXPECTED_DATA_REVISION = "cfb4b64d289acaf592237179f90dd23d3f2d5bdf"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/ruri_jamcqa_zero_shot.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encode(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            normalized = F.normalize(pooled.float(), dim=-1)
            if not torch.isfinite(normalized).all():
                raise ValueError("Ruri produced non-finite sentence embeddings")
            outputs.append(normalized.cpu())
    if not outputs:
        raise ValueError("cannot encode an empty text list")
    return torch.cat(outputs)


def _cosine_logits(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if query.ndim != 1 or candidates.ndim != 2 or candidates.shape[-1] != query.shape[-1]:
        raise ValueError("query and candidate vectors have incompatible shapes")
    return torch.mv(candidates, query)


def _group_metrics(
    logits: list[torch.Tensor], targets: list[int], groups: list[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    result = {}
    for group, indices in sorted(grouped.items()):
        result[group] = {
            **_metrics([logits[index] for index in indices], [targets[index] for index in indices]),
            "count": len(indices),
        }
    return result


def _mean_confidence(logits: list[torch.Tensor]) -> float:
    return float(torch.stack([torch.softmax(row.float(), dim=-1).max() for row in logits]).mean())


def main() -> None:
    args = _args()
    run_started_at = datetime.now(UTC).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["encoder"]["revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError("unexpected Ruri revision")
    if config["dataset"]["revision"] != EXPECTED_DATA_REVISION:
        raise ValueError("unexpected JamC-QA-V2 revision")
    if config["dataset"]["split"] != "dev" or config["dataset"].get("test_split_loaded", False):
        raise ValueError("only the JamC-QA-V2 dev split is permitted in this diagnostic")
    if (
        config["scoring"]["training"] != "none"
        or config["scoring"]["checkpoint_selection"] != "none"
    ):
        raise ValueError("this evaluator must remain zero-shot and must not select a checkpoint")

    data_path = Path(config["dataset"]["local_jsonl"])
    provenance_path = Path(config["dataset"]["provenance_file"])
    model_path = Path(config["encoder"]["local_path"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["revision"] != EXPECTED_DATA_REVISION or provenance["split"] != "dev":
        raise ValueError("local data provenance does not match the pinned dev split")
    if provenance["normalized_jsonl_sha256"] != sha256_file(data_path):
        raise ValueError("normalized JamC-QA-V2 dev data hash mismatch")
    expected_data_hash = "acd98fd9a1cf3fd5c59bb321b4480141fae6d124b70779599280171741ed3198"
    if provenance["parquet_sha256"] != expected_data_hash:
        raise ValueError("source Parquet hash mismatch")
    if sha256_file(model_path / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("local Ruri weights do not match the pinned model hash")

    rows = _load_rows(data_path)
    if len(rows) != int(config["dataset"]["expected_rows"]):
        raise ValueError("unexpected JamC-QA-V2 dev row count")
    if any(row.get("split") != "dev" or not row["id"].startswith("jamcqa-dev-") for row in rows):
        raise ValueError("non-dev or test question detected")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate question IDs detected")
    if any(len(row["options"]) != 4 or not 0 <= int(row["target_index"]) < 4 for row in rows):
        raise ValueError("unexpected option count or target index")

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configured local CUDA device is unavailable")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pooling_config = json.loads((model_path / "1_Pooling" / "config.json").read_text())
    if not pooling_config.get("pooling_mode_mean_tokens") or pooling_config.get(
        "pooling_mode_cls_token"
    ):
        raise ValueError("local Sentence-Transformers pooling config is not mean-token pooling")

    query_prefix = str(config["encoder"]["query_prefix"])
    candidate_prefix = str(config["encoder"]["candidate_prefix"])
    questions = [query_prefix + row["question"] for row in rows]
    options = [candidate_prefix + option for row in rows for option in row["options"]]
    started = time.perf_counter()
    question_vectors = _encode(
        questions,
        tokenizer,
        model,
        device,
        max_length=int(config["encoder"]["max_length"]),
        batch_size=int(config["encoder"]["batch_size"]),
    )
    option_vectors = _encode(
        options,
        tokenizer,
        model,
        device,
        max_length=int(config["encoder"]["max_length"]),
        batch_size=int(config["encoder"]["batch_size"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encoding_seconds = time.perf_counter() - started

    targets = []
    logits = []
    reversed_max_delta = 0.0
    offset = 0
    for row, query in zip(rows, question_vectors, strict=True):
        candidates = option_vectors[offset : offset + len(row["options"])]
        offset += len(row["options"])
        score = _cosine_logits(query, candidates)
        reversed_score = _cosine_logits(query, candidates.flip(0)).flip(0)
        reversed_max_delta = max(reversed_max_delta, float((score - reversed_score).abs().max()))
        targets.append(int(row["target_index"]))
        logits.append(score / float(config["scoring"]["softmax_temperature"]))
    if offset != len(option_vectors):
        raise AssertionError("encoded candidate vectors were not fully consumed")
    if reversed_max_delta > 1e-7:
        raise ValueError(
            f"option-order permutation changed aligned cosine scores: {reversed_max_delta}"
        )

    metrics = _metrics(logits, targets)
    categories = [str(row["category"]) for row in rows]
    by_category = _group_metrics(logits, targets, categories)
    macro_category = {
        name: sum(float(result[name]) for result in by_category.values()) / len(by_category)
        for name in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    probabilities = [torch.softmax(row.float(), dim=-1) for row in logits]
    predictions = []
    for row, score, probability in zip(rows, logits, probabilities, strict=True):
        predicted_index = int(score.argmax())
        predictions.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "options": row["options"],
                "target_index": int(row["target_index"]),
                "prediction_index": predicted_index,
                "target": row["target"],
                "prediction": row["options"][predicted_index],
                "cosine_logits": score.tolist(),
                "probabilities": probability.tolist(),
            }
        )
    predictions_path = output / "dev-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    group_counts = Counter(categories)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "schema_version": 1,
        "status": "complete_ruri_jamcqa_dev_zero_shot_diagnostic",
        "scope": "Japanese text-only zero-shot cosine ranking on the JamC-QA-V2 public dev split",
        "run_started_utc": run_started_at,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_id": "sbintuitions/JamC-QA-V2",
        "dataset_revision": EXPECTED_DATA_REVISION,
        "dataset_license": "CC-BY-SA-4.0",
        "source_parquet_sha256": provenance["parquet_sha256"],
        "normalized_dev_sha256": sha256_file(data_path),
        "fetch_report_sha256": sha256_file(provenance_path),
        "dev_ids_sha256": _sha256_text("\n".join(row["id"] for row in rows)),
        "count": len(rows),
        "category_counts": dict(sorted(group_counts.items())),
        "duplicate_option_rows_retained": provenance["duplicate_option_rows"],
        "model_id": "cl-nagoya/ruri-v3-30m",
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_weights_sha256": EXPECTED_MODEL_SHA256,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_json_sha256": sha256_file(model_path / "tokenizer.json"),
        "pooling_config_sha256": sha256_file(model_path / "1_Pooling" / "config.json"),
        "encoder_parameters": parameter_count,
        "encoder_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "precision": str(next(model.parameters()).dtype),
        "query_prefix": query_prefix,
        "candidate_prefix": candidate_prefix,
        "pooling": "Sentence-Transformers masked mean token pooling; L2-normalized",
        "score": "cosine / fixed_temperature_1.0; no calibration or learning",
        "checkpoint_selection": "none",
        "training_examples": 0,
        "test_split_loaded": False,
        "sealed_audit_loaded": False,
        "validation_metrics": {**metrics, "mean_confidence": _mean_confidence(logits)},
        "validation_by_category": by_category,
        "macro_category_metrics": macro_category,
        "option_order_max_abs_delta_after_alignment": reversed_max_delta,
        "option_order_stable": reversed_max_delta <= 1e-7,
        "timing_seconds": {"question_and_option_encoding": encoding_seconds},
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
        },
        "predictions_sha256": sha256_file(predictions_path),
        "predictions_count": len(predictions),
        "note": (
            "The 52-row dev split is small and intended by its authors for few-shot development. "
            "This frozen zero-shot check is not a general Japanese Decision quality claim; "
            "Ruri training-data contamination was not independently excluded."
        ),
    }
    report_path = output / "run-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
