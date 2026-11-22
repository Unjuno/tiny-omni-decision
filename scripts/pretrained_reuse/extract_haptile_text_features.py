"""Cache frozen MiniLM instruction and fixed-action option-pair features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_action import (  # noqa: E402
    ACTION_LABELS,
    ACTION_OPTION_TEXTS,
    ACTION_QUESTION,
)

MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MODEL_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
TEXT_WIDTH = 384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_features(context_vectors: np.ndarray, option_vectors: np.ndarray) -> np.ndarray:
    """Encode each question-option pair as q, o, |q-o|, q*o."""
    context = np.asarray(context_vectors, dtype=np.float32)
    options = np.asarray(option_vectors, dtype=np.float32)
    if (context.ndim != 2 or context.shape[1] != TEXT_WIDTH
            or options.shape != (len(ACTION_LABELS), TEXT_WIDTH)
            or not np.isfinite(context).all() or not np.isfinite(options).all()):
        raise ValueError("MiniLM context/options must be finite 384-wide features")
    question = np.broadcast_to(context[:, None, :], (len(context), len(options), TEXT_WIDTH))
    candidate = np.broadcast_to(options[None, :, :], question.shape)
    result = np.concatenate(
        (question, candidate, np.abs(question - candidate), question * candidate), axis=-1
    )
    if result.shape != (len(context), len(ACTION_LABELS), 4 * TEXT_WIDTH):
        raise ValueError("question-option features have an unexpected shape")
    return result.astype(np.float32, copy=False)


def encode_text_rows(
    examples: list[dict[str, Any]], model_dir: Path, *, device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if not examples or batch_size < 1:
        raise ValueError("text encoding requires rows and a positive batch size")
    paths = [row.get("path") for row in examples]
    if (any(not isinstance(path, str) or not path for path in paths)
            or len(paths) != len(set(paths))):
        raise ValueError("text examples require unique stable media paths")
    for row in examples:
        if (not isinstance(row.get("instruction"), str) or not row["instruction"].strip()
                or row.get("target") not in ACTION_LABELS
                or not isinstance(row.get("sha256"), str)):
            raise ValueError("example lacks its instruction, target or media hash")
    weight_path = model_dir / "model.safetensors"
    if sha256_file(weight_path) != MODEL_WEIGHTS_SHA256:
        raise ValueError("MiniLM weights differ from the frozen local revision")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the pinned PyTorch/Transformers environment") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(torch_device)
    if (encoder.config.hidden_size != TEXT_WIDTH
            or sum(parameter.numel() for parameter in encoder.parameters()) != 17_389_824):
        raise ValueError("MiniLM architecture differs from the frozen parameter contract")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    contexts = [
        f"Question: {ACTION_QUESTION}\nTask instruction: {row['instruction']}"
        for row in examples
    ]
    generic_context = f"Question: {ACTION_QUESTION}"
    context_without_instruction = [generic_context] * len(examples)
    option_texts = [f"Candidate action: {text}" for text in ACTION_OPTION_TEXTS]
    texts = contexts + context_without_instruction + option_texts
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], max_length=128, truncation=True,
                padding=True, return_tensors="pt",
            ).to(torch_device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.extend(row.cpu().numpy().astype(np.float32, copy=False) for row in pooled)
    count = len(examples)
    context_vectors = np.stack(vectors[:count])
    generic_context_vectors = np.stack(vectors[count:2 * count])
    option_vectors = np.stack(vectors[2 * count:])
    pairs = pair_features(context_vectors, option_vectors)
    generic_pairs = pair_features(generic_context_vectors, option_vectors)
    del encoder
    return context_vectors, pairs, generic_context_vectors, generic_pairs, {
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_parameters": 17_389_824,
        "requested_device": device,
        "wall_seconds": time.perf_counter() - started,
    }


def extract_text_features(
    examples_path: Path, output_dir: Path, model_dir: Path, *,
    split_name: str, device: str = "cpu", batch_size: int = 64,
) -> dict[str, object]:
    if split_name not in {"train", "validation", "calibration"}:
        raise ValueError("only train, validation and calibration are supported")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite text feature directory: {output_dir}")
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise ValueError("state/text example file must contain a row list")
    contexts, options, generic_contexts, generic_options, runtime = encode_text_rows(
        examples, model_dir, device=device, batch_size=batch_size
    )
    output_dir.mkdir(parents=True)
    cache_path = output_dir / "text-features.npz"
    np.savez_compressed(
        cache_path,
        sample_paths=np.asarray([row["path"] for row in examples], dtype=np.str_),
        media_sha256=np.asarray([row["sha256"] for row in examples], dtype=np.str_),
        targets=np.asarray([row["target"] for row in examples], dtype=np.str_),
        context_features=contexts,
        option_features=options,
        context_features_no_instruction=generic_contexts,
        option_features_no_instruction=generic_options,
    )
    result: dict[str, object] = {
        "format": f"haptile_{split_name}_frozen_text_features_v1",
        "status": "complete_frozen_text_feature_extraction",
        "split": split_name,
        "examples_sha256": sha256_file(examples_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "example_count": len(examples),
        "context_shape": list(contexts.shape),
        "option_shape": list(options.shape),
        "context_shape_no_instruction": list(generic_contexts.shape),
        "option_shape_no_instruction": list(generic_options.shape),
        "question_option_encoding": "q,o,abs(q-o),q*o",
        "pooling": "final_layer_attention_masked_mean",
        **runtime,
        "sealed_audit_opened": False,
        "quantization_performed": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(extract_text_features(
        args.examples, args.output_dir, args.model_dir,
        split_name=args.split, device=args.device, batch_size=args.batch_size,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
