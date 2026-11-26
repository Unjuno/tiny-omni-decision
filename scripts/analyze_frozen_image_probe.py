"""Read-only train/validation analysis for a cached CLEVR-4 image probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_image_probe import _option_embeddings, _predict, _tasks
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _uniform_baseline(count: int, choices: int = 10) -> dict[str, object]:
    return {
        "count": count,
        "expected_accuracy": 1 / choices,
        "nll": math.log(choices),
        "brier": 1 - 1 / choices,
        "ece_15_bins": 0.0,
    }


def _group_metrics(
    logits: list[torch.Tensor], targets: list[int], taxonomies: list[str]
) -> dict[str, dict[str, object]]:
    return {
        taxonomy: _metrics(
            [
                logit
                for logit, current in zip(logits, taxonomies, strict=True)
                if current == taxonomy
            ],
            [
                target
                for target, current in zip(targets, taxonomies, strict=True)
                if current == taxonomy
            ],
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }


def main() -> None:
    args = _args()
    sample_dir, run_dir = args.sample_dir.resolve(), args.run_dir.resolve()
    report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    features = torch.load(run_dir / "image-features.pt", map_location="cpu", weights_only=True)
    train_rows = _rows(sample_dir / "train-labels.jsonl")
    validation_rows = _rows(sample_dir / "val-labels.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.text_model, local_files_only=True).eval()
    option_embeddings, text_hidden = _option_embeddings(tokenizer, encoder, torch.device("cpu"))
    train_features, train_targets, _, train_taxonomies = _tasks(
        train_rows, features["train_features"], option_embeddings
    )
    validation_features, validation_targets, _, validation_taxonomies = _tasks(
        validation_rows, features["validation_features"], option_embeddings
    )
    scorer = CandidateScorer(768 + text_hidden)
    scorer.load_state_dict(
        torch.load(run_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    )
    train_logits = _predict(scorer, train_features, torch.device("cpu"))
    validation_logits = _predict(scorer, validation_features, torch.device("cpu"))
    validation_by_taxonomy = _group_metrics(
        validation_logits, validation_targets, validation_taxonomies
    )
    summary = {
        "schema_version": 1,
        "run_report_sha256": sha256_file(run_dir / "run-report.json"),
        "image_feature_cache_sha256": sha256_file(run_dir / "image-features.pt"),
        "selected_readout_sha256": sha256_file(run_dir / "best-readout.pt"),
        "validation_predictions_sha256": sha256_file(run_dir / "validation-predictions.jsonl"),
        "source_fetch_report_sha256": sha256_file(sample_dir / "fetch-report.json"),
        "train_task_count": len(train_targets),
        "validation_task_count": len(validation_targets),
        "train_metrics_at_selected_checkpoint": _metrics(train_logits, train_targets),
        "validation_metrics_at_selected_checkpoint": _metrics(
            validation_logits, validation_targets
        ),
        "validation_metrics_by_taxonomy": validation_by_taxonomy,
        "uniform_choice_baseline": _uniform_baseline(len(validation_targets)),
        "uniform_choice_baseline_by_taxonomy": {
            taxonomy: _uniform_baseline(sum(item == taxonomy for item in validation_taxonomies))
            for taxonomy in CLEVR4_TAXONOMIES
        },
        "selected_epoch": report["best_epoch"],
        "selection_rule": "minimum aggregate validation NLL over 8 recorded epochs",
        "training_loss_decreased_all_epochs": all(
            report["history"][index]["train_ce"] > report["history"][index + 1]["train_ce"]
            for index in range(len(report["history"]) - 1)
        ),
        "validation_sample_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in validation_rows).encode()
        ).hexdigest(),
    }
    result_path = run_dir / "posthoc-metrics.json"
    result_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
