"""Evaluate the frozen Social IQa MiniLM readout once on official publisher dev.

The script validates train/dev content disjointness before reading dev labels.
It has no fitting, calibration, prompt-search, or checkpoint-selection path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from train_socialiqa_frozen_minilm_readout import (
    CandidateScorer,
    _evaluate,
    _extract_features,
)
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.socialiqa_readout import (
    context_hash,
    validate_source_content_disjoint,
)

ARCHIVE_SHA256 = "ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3"
MODEL_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
READOUT_SHA256 = "066e943de2dbe493f6257c32c24b11c2566cd36507326c2d5bef7b868db33fa3"
CONFIG_SHA256 = "737e7c062f633c82a76022980bf03c5a282abc9e76772a56b486568f7dfdf5cc"
RUN_REPORT_SHA256 = "870bd0c41e45c5b51978cc5ac8c02ca69b3f2535928cd7c218243a0ead2bb072"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode("utf-8").splitlines()]


def audit_train_dev_content(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check content identity without requiring or reading either split's labels."""
    overlap = validate_source_content_disjoint(train_rows, dev_rows)
    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_unique_contexts": len({context_hash(str(row["context"])) for row in train_rows}),
        "dev_unique_contexts": len({context_hash(str(row["context"])) for row in dev_rows}),
        "overlap": overlap,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-questions", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    checks = {
        "archive": (args.archive, ARCHIVE_SHA256),
        "model_weights": (args.model_dir / "model.safetensors", MODEL_SHA256),
        "selected_readout": (args.readout, READOUT_SHA256),
        "training_config": (args.config, CONFIG_SHA256),
        "training_report": (args.run_report, RUN_REPORT_SHA256),
    }
    observed_hashes = {}
    for name, (path, expected_hash) in checks.items():
        observed = _sha256_file(path)
        if observed != expected_hash:
            raise ValueError(f"{name} hash mismatch: {observed}")
        observed_hashes[name] = observed

    with zipfile.ZipFile(args.archive) as source:
        train_member = "socialiqa-train-dev/train.jsonl"
        dev_member = "socialiqa-train-dev/dev.jsonl"
        dev_label_member = "socialiqa-train-dev/dev-labels.lst"
        train_bytes = source.read(train_member)
        dev_bytes = source.read(dev_member)
        # Deliberately defer access to dev-labels.lst until the exact-content
        # leakage gate has passed.
    train_source_rows = _read_jsonl(train_bytes)
    dev_source_rows = _read_jsonl(dev_bytes)
    split_audit = audit_train_dev_content(train_source_rows, dev_source_rows)
    if len(train_source_rows) != 33_410 or len(dev_source_rows) != 1_954:
        raise ValueError("unexpected pinned Social IQa train/dev row count")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    candidate_pairs = [
        (str(row["context"]), f"{row['question']} Answer option: {row[key]}")
        for row in dev_source_rows
        for key in ("answerA", "answerB", "answerC")
    ]
    lengths = tokenizer(
        [pair[0] for pair in candidate_pairs],
        [pair[1] for pair in candidate_pairs],
        truncation=False,
        padding=False,
    )["input_ids"]
    length_preflight = {
        "candidate_pairs": len(candidate_pairs),
        "max_untruncated_tokens": max(map(len, lengths), default=0),
        "would_truncate_at_256": sum(len(item) > 256 for item in lengths),
    }
    if length_preflight["would_truncate_at_256"]:
        raise ValueError("official dev would be truncated; refusing to read dev labels")
    print(
        json.dumps(
            {
                "phase": "content_and_length_preflight_passed_before_dev_labels",
                "split_audit": split_audit,
                "length_preflight": length_preflight,
            },
            indent=2,
        ),
        flush=True,
    )

    # Only after the split audit passed do we open publisher dev labels.
    with zipfile.ZipFile(args.archive) as source:
        label_bytes = source.read(dev_label_member)
    labels = [line.strip() for line in label_bytes.decode("utf-8").splitlines()]
    if len(labels) != len(dev_source_rows) or any(label not in {"1", "2", "3"} for label in labels):
        raise ValueError("publisher dev labels do not align with pinned rows")

    rows = []
    for index, (row, label) in enumerate(zip(dev_source_rows, labels, strict=True)):
        rows.append(
            {
                "sample_id": f"socialiqa-official-dev-{index:05d}",
                "context": str(row["context"]),
                "question": str(row["question"]),
                "options": [str(row[key]) for key in ("answerA", "answerB", "answerC")],
                "target": int(label) - 1,
                "source_context_sha256": context_hash(str(row["context"])),
            }
        )

    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    started_at = datetime.now(UTC).isoformat()
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to("cpu").eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    if encoder_parameters != 22_713_216 or any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        raise ValueError("MiniLM parameter/frozen-state check failed")

    feature_path = args.output_dir / "official-dev-features.pt"
    feature_meta = _extract_features(
        rows,
        tokenizer=tokenizer,
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        batch_questions=args.batch_questions,
        max_length=256,
        output_path=feature_path,
    )
    features = torch.load(feature_path, map_location="cpu", weights_only=True)
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    scorer = CandidateScorer(input_size=384, hidden_size=128, dropout=0.1).cpu()
    state = torch.load(args.readout, map_location="cpu", weights_only=True)
    scorer.load_state_dict(state, strict=True)
    metrics, logits, probabilities = _evaluate(scorer, features, targets)

    reloaded = CandidateScorer(input_size=384, hidden_size=128, dropout=0.1).cpu()
    reloaded.load_state_dict(
        torch.load(args.readout, map_location="cpu", weights_only=True), strict=True
    )
    _, reload_logits, reload_probs = _evaluate(reloaded, features, targets)
    reload_delta = max(
        float((logits - reload_logits).abs().max()),
        float((probabilities - reload_probs).abs().max()),
    )
    if reload_delta != 0.0:
        raise ValueError(f"readout reload changed test predictions: {reload_delta}")

    prediction_path = args.output_dir / "official-dev-predictions.jsonl"
    prediction_rows = []
    for row, probability in zip(rows, probabilities.tolist(), strict=True):
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_context_sha256": row["source_context_sha256"],
                "target": row["target"],
                "prediction": int(max(range(3), key=lambda option: probability[option])),
                "probabilities": probability,
            }
        )
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )

    report = {
        "status": "complete_one_shot_socialiqa_official_dev_no_selection",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "archive_sha256": observed_hashes["archive"],
            "train_member_sha256": _sha256_bytes(train_bytes),
            "dev_member_sha256": _sha256_bytes(dev_bytes),
            "dev_labels_sha256": _sha256_bytes(label_bytes),
            "source_revision": "8835ceb9141d7896d9d968634a9b21ae440e3ec5",
            "split_audit_before_labels": split_audit,
            "dev_label_counts": {str(i): labels.count(str(i)) for i in (1, 2, 3)},
            "project_sealed_audit_opened": False,
        },
        "model": {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "weights_sha256": observed_hashes["model_weights"],
            "encoder_parameters": encoder_parameters,
            "readout_sha256": observed_hashes["selected_readout"],
            "readout_parameters": 49_409,
            "total_parameters_before_quantization": encoder_parameters + 49_409,
            "checkpoint_source_run_sha256": observed_hashes["training_report"],
            "training_config_sha256": observed_hashes["training_config"],
            "selection": (
                "unchanged epoch-8 readout selected on internal train-only "
                "validation; no test selection"
            ),
        },
        "inference": {
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": args.cpu_threads,
            "candidate_pairs": feature_meta["candidate_pairs"],
            "truncated_pairs": feature_meta["truncated_pairs"],
            "max_length": feature_meta["max_length"],
            "length_preflight_before_labels": length_preflight,
            "feature_extraction_seconds": feature_meta["elapsed_seconds"],
            "readout_reload_max_probability_or_logit_delta": reload_delta,
        },
        "metrics": metrics,
        "predictions_sha256": _sha256_file(prediction_path),
        "feature_cache_sha256": feature_meta["feature_sha256"],
        "feature_cache_bytes": feature_meta["feature_bytes"],
        "evaluator_code_sha256": _sha256_file(Path(__file__)),
        "limitations": [
            "public dev is not a blind project sealed audit and must not be retuned or repeated",
            (
                "Social IQa has no stable ATOMIC source-event ID; exact context "
                "overlap is zero, but paraphrased source-event overlap remains unknown"
            ),
            (
                "English three-choice social commonsense task only; no multimodal "
                "or product-quality claim"
            ),
        ],
    }
    report_path = args.output_dir / "official-dev-audit-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "split_audit": split_audit,
                "metrics": metrics,
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
