"""One-shot audit of frozen typed-decision readouts on publisher test rows.

This entry point never fits/selects a checkpoint. It requires explicit test
input, checks state identity disjointness from source TRAIN, reloads one frozen
readout per encoder, and writes per-example predictions plus fixed metrics.
Targets are teacher-generated distributions, not human adjudications.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers
from compare_typed_decisions_ja_encoders import (
    CandidateScorer,
    _encode,
    _metrics,
    sha256_file,
)

from tiny_omni_decision.typed_decisions_ja import (
    normalized_state_sha256,
    parse_typed_decisions_ja_row,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw = value.partition("=")
    if not separator or not name or not raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--encoder", action="append", type=_named_path, required=True)
    parser.add_argument("--readout", action="append", type=_named_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def _load_test_groups(
    train_path: Path, test_path: Path
) -> tuple[list[Any], dict[str, int]]:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if set(train["split"].astype(str)) != {"train"}:
        raise ValueError("reference corpus must contain publisher TRAIN only")
    if set(test["split"].astype(str)) != {"test"}:
        raise ValueError("evaluation corpus must contain publisher TEST only")
    train_ids = set(train["id"].astype(str))
    test_ids = set(test["id"].astype(str))
    if len(train_ids) != len(train) or len(test_ids) != len(test):
        raise ValueError("state IDs must be unique within each source split")
    if train_ids & test_ids:
        raise ValueError("publisher state IDs overlap between TRAIN and TEST")

    train_states = set()
    for value in train["state"]:
        state = json.loads(value) if isinstance(value, str) else value
        train_states.add(normalized_state_sha256(state))
    groups = []
    seen_test_states: dict[str, str] = {}
    for row in test.to_dict(orient="records"):
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        state_hash = normalized_state_sha256(state)
        if state_hash in train_states:
            raise ValueError("normalized state content overlaps publisher TRAIN")
        prior_id = seen_test_states.setdefault(state_hash, str(row["id"]))
        if prior_id != str(row["id"]):
            raise ValueError("duplicate normalized state occurs under multiple TEST IDs")
        groups.extend(parse_typed_decisions_ja_row(row, expected_split="test"))

    return groups, {
        "train_state_rows": len(train),
        "test_state_rows": len(test),
        "train_state_ids": len(train_ids),
        "test_state_ids": len(test_ids),
        "train_normalized_states": len(train_states),
        "test_normalized_states": len(seen_test_states),
        "train_test_id_overlap": 0,
        "train_test_normalized_state_overlap": 0,
        "test_decisions": len(groups),
    }


def _test_arrays(features: torch.Tensor, groups: list[Any]) -> tuple[torch.Tensor, ...]:
    max_options = max(len(group.option_labels) for group in groups)
    X = torch.zeros((len(groups), max_options, int(features.shape[-1])), dtype=torch.float32)
    Y = torch.zeros((len(groups), max_options), dtype=torch.float32)
    M = torch.zeros((len(groups), max_options), dtype=torch.bool)
    cursor = 0
    for index, group in enumerate(groups):
        count = len(group.option_labels)
        X[index, :count] = features[cursor : cursor + count]
        Y[index, :count] = torch.tensor(group.target_probabilities)
        M[index, :count] = True
        cursor += count
    if cursor != len(features):
        raise ValueError("candidate feature count does not match test options")
    return X, Y, M


def main() -> None:
    args = _parse_args()
    encoders = dict(args.encoder)
    readouts = dict(args.readout)
    if set(encoders) != set(readouts) or len(encoders) != len(args.encoder):
        raise ValueError("each unique encoder name must have exactly one readout")
    if not args.test_parquet.is_file() or not args.train_parquet.is_file():
        raise FileNotFoundError("pinned train and test parquet inputs are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now(UTC).isoformat()
    groups, split_audit = _load_test_groups(args.train_parquet, args.test_parquet)
    report: dict[str, Any] = {
        "status": "running_one_shot_publisher_test_no_selection",
        "started_at_utc": started,
        "dataset_revision": args.dataset_revision,
        "data": {
            "train_parquet_sha256": sha256_file(args.train_parquet),
            "test_parquet_sha256": sha256_file(args.test_parquet),
            "split_audit": split_audit,
        },
        "protocol": {
            "checkpoint_selection": "none; frozen validation-selected readouts loaded as-is",
            "metric_targets": "publisher English-state teacher distributions; not human truth",
            "max_length": args.max_length,
            "truncation": "only_first",
            "encoder_device": "cpu",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": args.cpu_threads,
        },
        "candidates": {},
    }

    for name, model_dir in encoders.items():
        if name not in {"bert-base", "light-splade"}:
            raise ValueError("candidate names must be the predeclared frozen pair")
        readout_path = readouts[name]
        if not readout_path.is_file():
            raise FileNotFoundError(readout_path)
        started_model = time.perf_counter()
        features, _, encoder_meta = _encode(
            model_dir,
            args.tokenizer_dir,
            groups,
            encode_batch_size=args.encode_batch_size,
            max_length=args.max_length,
            cpu_threads=args.cpu_threads,
            trust_remote_tokenizer_code=False,
        )
        X, Y, M = _test_arrays(features, groups)
        scorer = CandidateScorer(int(features.shape[-1]))
        state = torch.load(readout_path, map_location="cpu", weights_only=True)
        scorer.load_state_dict(state, strict=True)
        scorer.eval()
        with torch.inference_mode():
            logits = scorer(X).masked_fill(~M, -1e9)
            metrics = _metrics(logits, Y, M, groups)
        predictions_path = args.output_dir / f"{name}-test-predictions.jsonl"
        predictions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in metrics["predictions"]
            ),
            encoding="utf-8",
        )
        report["candidates"][name] = {
            "encoder_dir": str(model_dir.resolve()),
            "encoder_config": str(model_dir / "config.json"),
            "encoder_config_sha256": sha256_file(model_dir / "config.json"),
            "readout_path": str(readout_path.resolve()),
            "readout_sha256": sha256_file(readout_path),
            "encoder_metadata": encoder_meta,
            "test_metrics": {key: value for key, value in metrics.items() if key != "predictions"},
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "evaluation_seconds": time.perf_counter() - started_model,
        }
        del features, X, Y, M, scorer, state

    report["ended_at_utc"] = datetime.now(UTC).isoformat()
    report["status"] = "complete_one_shot_publisher_test_no_selection"
    report_path = args.output_dir / "test-audit-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "split_audit": split_audit,
        "metrics": {
            name: result["test_metrics"]["macro_over_workflow_question_type_cells"]
            for name, result in report["candidates"].items()
        },
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
