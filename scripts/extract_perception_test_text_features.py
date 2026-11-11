"""Cache pinned MiniLM question/candidate features for frozen Path A."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
TEXT_PARAMETERS = 17_389_824
EMBEDDING_SIZE = 384
MAX_LENGTH = 256
OPTION_PREFIX = re.compile(r"^[A-C]:\s*(.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_option_prefix(option: str) -> str:
    if not isinstance(option, str):
        raise ValueError("each candidate must be a string")
    match = OPTION_PREFIX.fullmatch(option.strip())
    if match is None or not match.group(1).strip():
        raise ValueError("candidate must have one A/B/C label followed by nonempty text")
    return match.group(1).strip()


def candidate_prompts(row: dict[str, Any]) -> list[str]:
    question = row.get("question")
    options = row.get("options")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    if not isinstance(options, list) or len(options) != 3:
        raise ValueError("exactly three candidates are required")
    candidates = [strip_option_prefix(option) for option in options]
    return [f"Question: {question.strip()}\nCandidate answer: {option}" for option in candidates]


def expected_source_split(split: str) -> str:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    return "train" if split == "train" else "valid"


def _read_rows(corpus_dir: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "r2-perception-test-mcqa-path-a-20261111":
        raise ValueError("unexpected frozen corpus manifest")
    data = manifest.get(split)
    if not isinstance(data, dict):
        raise ValueError("corpus manifest lacks requested split")
    rows_path = corpus_dir / f"{split}.jsonl"
    if sha256_file(rows_path) != data.get("records_sha256"):
        raise ValueError("frozen JSONL hash differs from corpus manifest")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != data.get("record_count"):
        raise ValueError("frozen JSONL count differs from corpus manifest")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs in frozen JSONL")
    expected_split = expected_source_split(split)
    if any(row.get("split") != expected_split for row in rows):
        raise ValueError("frozen JSONL has an unexpected split value")
    return rows, manifest


def extract_text_features(
    *,
    corpus_dir: Path,
    model_dir: Path,
    output_dir: Path,
    split: str,
    batch_size: int = 64,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if batch_size < 1 or device_name not in {"cpu", "cuda"}:
        raise ValueError("positive batch size and cpu/cuda device are required")
    corpus_dir = corpus_dir.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    weight_path = model_dir / "model.safetensors"
    if sha256_file(weight_path) != TEXT_WEIGHT_SHA256:
        raise ValueError("MiniLM weight hash mismatch")
    rows, corpus_manifest = _read_rows(corpus_dir, split)
    prompts = [prompt for row in rows for prompt in candidate_prompts(row)]

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    device = torch.device(device_name)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TEXT_PARAMETERS:
        raise ValueError(f"unexpected MiniLM parameter count: {parameter_count}")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = pooled.float()
            if pooled.shape[1] != EMBEDDING_SIZE or not torch.isfinite(pooled).all():
                raise FloatingPointError("MiniLM returned invalid candidate embeddings")
            chunks.append(pooled.cpu().numpy())
    features = np.concatenate(chunks, axis=0).reshape(len(rows), 3, EMBEDDING_SIZE)
    if not np.isfinite(features).all():
        raise FloatingPointError("text feature cache contains non-finite values")

    output_dir = output_dir.resolve()
    if output_dir.drive.upper() != "D:":
        raise ValueError("text feature outputs must stay on D:")
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / f"{split}-candidate-features.npz"
    with feature_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=np.asarray([row["id"] for row in rows]),
            video_ids=np.asarray([row["source_video_id"] for row in rows]),
            features=features,
        )
        stream.flush()

    model_files = {
        path.name: sha256_file(path) for path in sorted(model_dir.iterdir()) if path.is_file()
    }
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "split": split,
        "records_sha256": corpus_manifest[split]["records_sha256"],
        "record_count": len(rows),
        "sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode("utf-8")
        ).hexdigest(),
        "model_revision": TEXT_REVISION,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "command_line": sys.argv,
        "python_version": platform.python_version(),
        "model_parameters": parameter_count,
        "model_files_sha256": model_files,
        "prompt_template": (
            "Question: {question}\nCandidate answer: {option_text_without_A_B_C_prefix}"
        ),
        "option_prefix_removed_before_encoding": True,
        "rows_with_duplicate_candidate_text_ids": [
            row["id"] for row in rows if len(set(candidate_prompts(row))) < 3
        ],
        "duplicate_candidate_rows_excluded": 0,
        "pooling": "masked mean of last_hidden_state",
        "max_length": MAX_LENGTH,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_file": feature_path.name,
        "feature_sha256": sha256_file(feature_path),
        "feature_bytes": feature_path.stat().st_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "labels_used": False,
        "official_test_opened": False,
    }
    (output_dir / "feature-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = extract_text_features(
        corpus_dir=args.corpus_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
