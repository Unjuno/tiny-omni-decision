"""Extract fixed Sentence-Transformers mean-pool embeddings for RoboPro prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
PARAMETERS = 17_389_824
MAX_LENGTH = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        raise ValueError("expected hidden [batch,tokens,dim] and mask [batch,tokens]")
    denominator = weights.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("every prompt must contain at least one unmasked token")
    pooled = (values * weights[..., None]).sum(axis=1) / denominator
    if not np.isfinite(pooled).all():
        raise ValueError("pooled prompt embeddings must be finite")
    return pooled


def extract(sample_path: Path, model_dir: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite text-feature output: {output_root}")
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    if sample["revision"] != "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0":
        raise ValueError("unexpected RoboPro source revision")
    weights_path = model_dir / "model.safetensors"
    if sha256_file(weights_path) != WEIGHTS_SHA256:
        raise ValueError("pinned MiniLM weight SHA-256 mismatch")
    modules = json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))
    pooling = json.loads((model_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    if (
        len(modules) != 2
        or modules[1]["type"] != "sentence_transformers.models.Pooling"
        or not pooling.get("pooling_mode_mean_tokens")
        or any(
            pooling.get(key, False)
            for key in (
                "pooling_mode_cls_token",
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
            )
        )
    ):
        raise ValueError("unexpected Sentence-Transformers pooling contract")
    prompts = []
    prompt_hashes = []
    split_rows = []
    for split in ("train", "development"):
        for group in sample[split]:
            prompts.append(str(group["prompt"]))
            prompt_hashes.append(str(group["prompt_sha256"]))
            split_rows.append(split)

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
        )
        .cpu()
        .eval()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PARAMETERS:
        raise ValueError(f"unexpected MiniLM parameter count: {parameter_count}")
    all_features = []
    with torch.inference_mode():
        for offset in range(0, len(prompts), 16):
            batch = tokenizer(
                prompts[offset : offset + 16],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state.cpu().numpy()
            features = masked_mean_pool(hidden, batch["attention_mask"].cpu().numpy())
            all_features.append(features)
    embedding = np.concatenate(all_features).astype(np.float32)
    if embedding.shape != (len(prompts), 384) or not np.isfinite(embedding).all():
        raise ValueError(f"invalid prompt feature shape or values: {embedding.shape}")

    output_root.mkdir(parents=True, exist_ok=False)
    cache_path = output_root / "minilm-prompt-features.npz"
    np.savez_compressed(
        cache_path,
        prompt=np.asarray(prompts),
        prompt_sha256=np.asarray(prompt_hashes),
        split=np.asarray(split_rows),
        features=embedding,
    )
    report = {
        "schema_version": "robopro-minilm-prompt-features-v1",
        "status": "feature_extraction_pass",
        "source_revision": sample["revision"],
        "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_sha256": sha256_file(weights_path),
        "frozen_parameters": parameter_count,
        "pooling": "un-normalized attention-mask mean of last hidden state",
        "max_length": MAX_LENGTH,
        "prompt_count": len(prompts),
        "train_prompt_count": sum(split == "train" for split in split_rows),
        "development_prompt_count": sum(split == "development" for split in split_rows),
        "feature_shape": list(embedding.shape),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cpu_only": not torch.cuda.is_initialized(),
        "wall_seconds": time.perf_counter() - started,
        "cache": {
            "path": cache_path.name,
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
        },
        "official_evaluation_or_audit_loaded": False,
    }
    report_path = output_root / "prompt-feature-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_parameters": parameter_count,
                "features": report["feature_shape"],
                "cache": report["cache"],
                "wall_seconds": report["wall_seconds"],
                "report_sha256": report_sha,
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    extract(
        args.sample_manifest.resolve(strict=True),
        args.model_dir.resolve(strict=True),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
