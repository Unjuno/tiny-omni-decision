"""Extract pinned CPU MiniLM features for the frozen Clotho publisher cohort."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.fetch_clotho_publisher_val_media import ATTEMPT, MANIFEST_SHA256, load_manifest
from tiny_omni_decision.audio_runtime import TEXT_MODEL_REVISION, TEXT_WEIGHTS_SHA256

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUTPUT = ATTEMPT / "text-features"
TEXT_ROOT = ROOT / "models/text-minilm-en"
PARITY_COUNT = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(model, tokenizer, questions: list[str]) -> tuple[np.ndarray, int]:
    batches, max_tokens = [], 0
    for start in range(0, len(questions), 32):
        tokens = tokenizer(
            questions[start : start + 32],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        max_tokens = max(max_tokens, int(tokens["attention_mask"].sum(1).max()))
        if tokens["input_ids"].shape[1] > 128:
            raise ValueError("question exceeds 128-token limit")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            features = (hidden * mask).sum(1) / mask.sum(1)
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite MiniLM feature")
        batches.append(features.cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False), max_tokens


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    questions = load_manifest()
    weights = TEXT_ROOT / "model.safetensors"
    if sha256(weights) != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM weights hash mismatch")
    model = (
        AutoModel.from_pretrained(TEXT_ROOT, local_files_only=True)
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ROOT, local_files_only=True)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("text features must be extracted on CPU")

    started = time.perf_counter()
    questions_text = [row["question"] for row in questions]
    features, max_tokens = encode(model, tokenizer, questions_text)
    old_text_dir = ROOT / "clotho-aqa-question-features-01"
    old_report = json.loads((old_text_dir / "result.json").read_text(encoding="utf-8"))
    old_path = old_text_dir / "validation.npy"
    if sha256(old_path) != old_report["splits"]["validation"]["feature_sha256"]:
        raise ValueError("frozen development MiniLM cache hash mismatch")
    validation_rows = json.loads(
        (ROOT / "clotho-aqa-development-split-01" / "validation.json").read_text(
            encoding="utf-8"
        )
    )[:PARITY_COUNT]
    replay, _ = encode(model, tokenizer, [row["question"] for row in validation_rows])
    old_features = np.load(old_path, allow_pickle=False)
    parity_error = float(np.max(np.abs(replay - old_features[:PARITY_COUNT])))
    if parity_error > 1e-6:
        raise ValueError("MiniLM feature preprocessing differs from frozen reference")

    output.mkdir(parents=True, exist_ok=False)
    feature_path = output / "question-features.npy"
    np.save(feature_path, features)
    ids = [[row["source_id"], row["normalized_question"]] for row in questions]
    ids_bytes = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    (output / "question-ids.json").write_bytes(ids_bytes)
    model_files = {
        str(path.relative_to(TEXT_ROOT)): sha256(path)
        for path in TEXT_ROOT.rglob("*")
        if path.is_file()
    }
    report = {
        "status": "PASS",
        "scope": "frozen text features only; no fitting or calibration",
        "device": "cpu",
        "torch": torch.__version__,
        "manifest_sha256": MANIFEST_SHA256,
        "encoder": {
            "revision": TEXT_MODEL_REVISION,
            "weights_sha256": TEXT_WEIGHTS_SHA256,
            "local_file_sha256": model_files,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "shape": list(features.shape),
            "max_tokens": max_tokens,
            "pooling": "attention-mask mean of final hidden states",
            "parity_source": (
                "first 8 fixed rows of existing Clotho development validation features"
            ),
            "parity_max_abs_error": parity_error,
        },
        "question_count": len(questions),
        "ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "feature_sha256": sha256(feature_path),
        "seconds": time.perf_counter() - started,
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
