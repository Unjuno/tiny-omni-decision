"""Read-only completion metrics and uniform-choice baseline for a text probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
import yaml

from scripts.train_frozen_text_probe import CandidateScorer, _metrics, _predict
from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl, sha256_file
from tiny_omni_decision.schema import DatasetManifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _id_hash(examples: list[object]) -> str:
    return hashlib.sha256("\n".join(item.id for item in examples).encode()).hexdigest()


def _uniform_baseline(option_counts: list[int], bins: int = 15) -> dict[str, object]:
    n = len(option_counts)
    confidences = [1.0 / count for count in option_counts]
    bin_counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bin_counts.append(
            sum(
                low <= confidence < high if index < bins - 1 else low <= confidence <= high
                for confidence in confidences
            )
        )
    return {
        "count": n,
        "expected_accuracy": sum(confidences) / n,
        "nll": sum(math.log(count) for count in option_counts) / n,
        "brier": sum(1 - confidence for confidence in confidences) / n,
        "ece_15_bins": 0.0,
        "ece_bin_counts": bin_counts,
    }


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    )
    train = list(
        normalize_jsonl(
            iter_local_rows(args.train),
            manifest.model_copy(update={"split": "train"}),
            "typed-decisions-synth",
        )
    )
    validation = list(
        normalize_jsonl(
            iter_local_rows(args.validation),
            manifest.model_copy(update={"split": "validation"}),
            "typed-decisions-synth",
        )
    )
    artifact = torch.load(run_dir / "frozen-features.pt", map_location="cpu", weights_only=True)
    state = torch.load(run_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    scorer = CandidateScorer(384 * 4)
    scorer.load_state_dict(state)
    train_logits = _predict(scorer, artifact["train_features"], torch.device("cpu"))
    validation_logits = _predict(scorer, artifact["validation_features"], torch.device("cpu"))
    train_metrics = _metrics(train_logits, artifact["train_targets"])
    validation_metrics = _metrics(validation_logits, artifact["validation_targets"])
    option_counts_train = [len(example.options) for example in train]
    option_counts_validation = [len(example.options) for example in validation]
    summary = {
        "schema_version": 1,
        "run_report_sha256": sha256_file(run_dir / "run-report.json"),
        "best_readout_sha256": sha256_file(run_dir / "best-readout.pt"),
        "train_sample_id_order_sha256": _id_hash(train),
        "validation_sample_id_order_sha256": _id_hash(validation),
        "train_option_count_distribution": dict(sorted(Counter(option_counts_train).items())),
        "validation_option_count_distribution": dict(
            sorted(Counter(option_counts_validation).items())
        ),
        "uniform_choice_validation_baseline": _uniform_baseline(option_counts_validation),
        "best_checkpoint_train_metrics": train_metrics,
        "best_checkpoint_validation_metrics": validation_metrics,
        "selected_epoch": report["best_epoch"],
        "training_loss_falls_after_best": bool(
            report["history"][-1]["train_ce"]
            < report["history"][report["best_epoch"] - 1]["train_ce"]
        ),
        "validation_nll_after_best": report["history"][-1]["validation_nll"],
        "validation_ece_after_best": report["history"][-1]["validation_ece_15_bins"],
    }
    out = run_dir / "posthoc-metrics.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
