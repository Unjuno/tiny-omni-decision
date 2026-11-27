"""Run one frozen, no-selection evaluation on pinned BANKING77 publisher test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from scripts.audit_banking77_train_quality import normalize_text
from scripts.run_banking77_minilm_probe import (
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
    _metrics,
    _pool_features,
)

EXPECTED_TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
EXPECTED_TEST_ROWS = 3080
SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_once(
    train_csv: Path,
    test_csv: Path,
    model_dir: Path,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if sha256(test_csv) != EXPECTED_TEST_SHA256:
        raise ValueError("BANKING77 publisher test SHA-256 mismatch")
    if sha256(train_csv) != result["data_sha256"]:
        raise ValueError("train data does not match the selected model receipt")
    readout_path = run_dir / result["readout_file"]
    if sha256(readout_path) != result["readout_sha256"]:
        raise ValueError("readout hash mismatch")
    model_dir = model_dir.resolve()
    if model_dir.name != MODEL_REVISION:
        raise ValueError("MiniLM revision mismatch")
    if sha256(model_dir / "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise ValueError("MiniLM weight hash mismatch")

    train_rows = list(
        csv.DictReader(
            io.StringIO(train_csv.read_text(encoding="utf-8-sig"), newline="")
        )
    )
    test_text = test_csv.read_text(encoding="utf-8-sig")
    test_rows = list(csv.DictReader(io.StringIO(test_text, newline="")))
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"expected {EXPECTED_TEST_ROWS} test rows, got {len(test_rows)}")
    if not test_rows or set(test_rows[0]) != {"text", "category"}:
        raise ValueError("expected the publisher's text,category CSV schema")

    labels = sorted({row["category"] for row in train_rows})
    if set(labels) != {row["category"] for row in test_rows}:
        raise ValueError("test labels differ from the frozen 77-intent vocabulary")
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_normalized = {normalize_text(row["text"]) for row in train_rows}
    exact_normalized_overlap = sum(
        normalize_text(row["text"]) in train_normalized for row in test_rows
    )

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True
    ).eval().to("cpu")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    features = _pool_features(
        [row["text"] for row in test_rows], tokenizer, encoder, batch_size=64
    )
    targets = torch.tensor(
        [label_to_id[row["category"]] for row in test_rows], dtype=torch.long
    )
    head = torch.nn.Linear(features.shape[-1], len(labels))
    head.load_state_dict(load_file(readout_path, device="cpu"))
    head.eval()
    with torch.no_grad():
        logits = head(features)
    metrics = _metrics(logits, targets, len(labels))
    predictions = metrics.pop("predictions")
    probabilities = metrics.pop("probabilities")

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "official-test-predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "test_row_id": row_id,
                    "target_intent": test_rows[row_id]["category"],
                    "predicted_intent": labels[predictions[row_id]],
                    "probabilities": probabilities[row_id],
                }
                for row_id in range(len(test_rows))
            ],
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "experiment_id": result["experiment_id"],
        "evaluation": "one-shot frozen official publisher test; no checkpoint selection or tuning",
        "publisher_test_revision": SOURCE_REVISION,
        "publisher_test_sha256": sha256(test_csv),
        "publisher_test_rows": len(test_rows),
        "train_sha256": sha256(train_csv),
        "selected_readout_sha256": sha256(readout_path),
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "normalized_exact_text_overlap_count_with_train": exact_normalized_overlap,
        "normalized_exact_text_overlap_rate": exact_normalized_overlap / len(test_rows),
        "test_intent_counts": dict(sorted(Counter(row["category"] for row in test_rows).items())),
        "metrics": metrics,
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256(predictions_path),
        "predictions_bytes": predictions_path.stat().st_size,
        "device": "cpu",
        "gpu_used": False,
        "labels_modified_or_rows_filtered": False,
        "project_sealed_audit_opened": False,
        "test_status": "now exposed; do not use for model selection or tuning",
        "limitations": [
            "BANKING77 has no user/session IDs for source-group bootstrap or "
            "customer-disjoint testing",
            "fine-grained labels have documented overlap and one unresolved train conflict",
            "this English banking-intent result does not establish arbitrary-option "
            "or multimodal Decision quality",
        ],
    }
    (output_dir / "official-test-result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = evaluate_once(
        args.train_csv,
        args.test_csv,
        args.model_dir,
        args.run_dir,
        args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
