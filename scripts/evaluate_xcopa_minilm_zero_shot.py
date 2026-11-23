"""Run a fixed zero-shot multilingual MiniLM diagnostic on XCOPA validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DATA_REVISION = "e2e9d7f105a758ee869cd13e0fe251ef93a29840"
DATA_BLOBS = {
    "et": "45bee3404d7dd9c94f4502a24d78eca5a3611034",
    "id": "4bb559c26c19da9d571e1603b4227d5d2d8e81d8",
    "it": "c7da0e9cb7802a0f242ded4e07b27ab5c0e14b7d",
    "th": "027e94ca4bdeca734b56475e335a501f6acd7e90",
    "tr": "206591c6c26ab5f222611962bb8ebfb6fb67bcc2",
    "vi": "237a41dbc54ad35c2e50ccfc7b3b04ba5fa391b6",
    "zh": "8c940ea5b620eeb3565df47b93a285257ca50efd",
}
LANGUAGE_NAMES = {
    "et": "Estonian",
    "id": "Indonesian",
    "it": "Italian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/xcopa_minilm_zero_shot.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _load_rows(path: Path, language: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 100:
        raise ValueError(f"expected 100 XCOPA validation rows for {language}, got {len(rows)}")
    expected_fields = {"premise", "choice1", "choice2", "question", "label", "idx", "changed"}
    if any(set(row) != expected_fields for row in rows):
        raise ValueError(f"unexpected XCOPA row schema for {language}")
    if any(row["question"] not in {"cause", "effect"} for row in rows):
        raise ValueError(f"unexpected causal question type for {language}")
    if any(int(row["label"]) not in {0, 1} for row in rows):
        raise ValueError(f"unexpected target label for {language}")
    if [int(row["idx"]) for row in rows] != list(range(100)):
        raise ValueError(f"unexpected row order or IDs for {language}")
    return rows


def _encode(
    texts: list[str], tokenizer: Any, model: Any, device: torch.device, *, batch_size: int
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            normalized = F.normalize(pooled.float(), dim=-1)
            if not torch.isfinite(normalized).all():
                raise ValueError("MiniLM produced non-finite sentence embeddings")
            outputs.append(normalized.cpu())
    if not outputs:
        raise ValueError("cannot encode an empty text list")
    return torch.cat(outputs)


def _versions() -> dict[str, str]:
    names = ("torch", "transformers", "huggingface-hub", "safetensors", "PyYAML")
    return {name: importlib.metadata.version(name) for name in names}


def _metrics(logits: list[torch.Tensor], targets: list[int], bins: int = 15) -> dict[str, Any]:
    if not logits:
        return {
            "count": 0,
            "accuracy": None,
            "nll": None,
            "brier": None,
            "ece_15_bins": None,
        }
    probabilities = [torch.softmax(item.float(), dim=-1) for item in logits]
    correct = torch.tensor(
        [
            int(int(prob.argmax()) == target)
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    nll = torch.tensor(
        [
            -float(prob[target].clamp_min(1e-12).log())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    brier = torch.tensor(
        [
            float(
                (prob - F.one_hot(torch.tensor(target), prob.numel()).to(prob.dtype)).square().sum()
            )
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([float(prob.max()) for prob in probabilities])
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        count = int(mask.sum())
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
    }


def main() -> None:
    args = _args()
    started_at = datetime.now(UTC).isoformat()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["dataset"]["revision"] != DATA_REVISION or config["dataset"]["split"] != "validation":
        raise ValueError("only the pinned XCOPA validation split may be evaluated")
    if config["encoder"]["revision"] != MODEL_REVISION:
        raise ValueError("unexpected MiniLM revision")
    if (
        config["scoring"]["training"] != "none"
        or config["scoring"]["checkpoint_selection"] != "none"
    ):
        raise ValueError("the evaluator must remain zero-shot without checkpoint selection")
    if float(config["scoring"]["temperature"]) != 0.05:
        raise ValueError("temperature is frozen at the Sentence-Transformers cosine scale 20")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_dir = Path(config["dataset"]["local_data_dir"])
    model_path = Path(config["encoder"]["local_path"])
    pooling_path = model_path / "1_Pooling" / "config.json"
    pooling = json.loads(pooling_path.read_text(encoding="utf-8"))
    if not pooling.get("pooling_mode_mean_tokens") or pooling.get("pooling_mode_cls_token"):
        raise ValueError("pinned model must use masked mean pooling")
    if int(config["encoder"]["max_length"]) != 128:
        raise ValueError("maximum sequence length is frozen at the model-card value 128")

    rows_by_language: dict[str, list[dict[str, Any]]] = {}
    data_hashes = {}
    for language, expected_blob in DATA_BLOBS.items():
        path = data_dir / f"val.{language}.jsonl"
        raw = path.read_bytes()
        actual_blob = _git_blob_sha(raw)
        if actual_blob != expected_blob:
            raise ValueError(f"pinned XCOPA blob mismatch for {language}: {actual_blob}")
        rows_by_language[language] = _load_rows(path, language)
        data_hashes[language] = {"git_blob_sha1": actual_blob, "sha256": _sha256(path)}

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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["encoder"]["expected_parameters"]):
        raise ValueError(f"unexpected MiniLM parameter count: {parameter_count}")

    language_results = {}
    all_predictions = []
    started = time.perf_counter()
    for language, rows in rows_by_language.items():
        premises = [row["premise"] for row in rows]
        choices = [choice for row in rows for choice in (row["choice1"], row["choice2"])]
        premise_vectors = _encode(
            premises, tokenizer, model, device, batch_size=int(config["encoder"]["batch_size"])
        )
        choice_vectors = _encode(
            choices, tokenizer, model, device, batch_size=int(config["encoder"]["batch_size"])
        )
        logits = []
        targets = []
        for index, row in enumerate(rows):
            candidate_vectors = choice_vectors[index * 2 : index * 2 + 2]
            scores = torch.mv(candidate_vectors, premise_vectors[index]) / float(
                config["scoring"]["temperature"]
            )
            permuted = torch.mv(candidate_vectors.flip(0), premise_vectors[index]).flip(0) / float(
                config["scoring"]["temperature"]
            )
            if not torch.allclose(scores, permuted, atol=1e-6, rtol=0):
                raise ValueError("option permutation changed aligned cosine scores")
            logits.append(scores)
            targets.append(int(row["label"]))
            prediction = int(scores.argmax())
            all_predictions.append(
                {
                    "language": language,
                    "idx": int(row["idx"]),
                    "question_type": row["question"],
                    "target_index": int(row["label"]),
                    "prediction_index": prediction,
                    "logits": scores.tolist(),
                    "probabilities": torch.softmax(scores, dim=-1).tolist(),
                }
            )
        language_results[language] = {
            "name": LANGUAGE_NAMES[language],
            **_metrics(logits, targets),
            "count": len(rows),
            "by_question_type": {
                question_type: {
                    **_metrics(
                        [
                            logits[i]
                            for i, row in enumerate(rows)
                            if row["question"] == question_type
                        ],
                        [
                            targets[i]
                            for i, row in enumerate(rows)
                            if row["question"] == question_type
                        ],
                    ),
                    "count": sum(row["question"] == question_type for row in rows),
                }
                for question_type in ("cause", "effect")
            },
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started

    language_names = list(language_results)
    macro = {
        metric: sum(float(language_results[language][metric]) for language in language_names)
        / len(language_names)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    (output / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in all_predictions),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "complete_xcopa_validation_zero_shot_diagnostic",
        "scope": (
            "fixed multilingual text-only cosine ranking on XCOPA validation; "
            "no training or test split access"
        ),
        "run_started_utc": started_at,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "dataset": {
            "repository": "cambridgeltl/xcopa",
            "revision": DATA_REVISION,
            "split": "validation",
            "license": "CC-BY-4.0",
            "languages": data_hashes,
            "excluded_languages": config["dataset"]["excluded_languages"],
        },
        "model": {
            "id": config["encoder"]["model_id"],
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "parameters": parameter_count,
            "model_safetensors_sha256": _sha256(model_path / "model.safetensors"),
            "pooling_config_sha256": _sha256(pooling_path),
            "pooling": "masked mean",
        },
        "scoring": config["scoring"],
        "count": len(all_predictions),
        "macro_language_metrics": macro,
        "by_language": language_results,
        "inference_seconds": inference_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": _versions(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"status": report["status"], "count": report["count"], "macro": macro}, indent=2)
    )


if __name__ == "__main__":
    main()
