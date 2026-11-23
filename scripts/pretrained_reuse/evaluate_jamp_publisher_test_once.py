"""One-shot Jamp publisher-test evaluation for validation-frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from scripts.pretrained_reuse.build_jamp_grouped_split import (
        LABELS,
        normalize_sentence,
    )
    from scripts.pretrained_reuse.finetune_jamp_distilbert_top_layers import (
        _encode_logits,
        _save_predictions,
        _tokenize,
    )
    from scripts.pretrained_reuse.train_jamp_frozen_text_readout import (
        LABEL_TO_ID,
        Readout,
        encode_features,
        metrics,
        predict,
        read_tsv,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:
    from build_jamp_grouped_split import LABELS, normalize_sentence
    from finetune_jamp_distilbert_top_layers import (
        _encode_logits,
        _save_predictions,
        _tokenize,
    )
    from train_jamp_frozen_text_readout import (
        LABEL_TO_ID,
        Readout,
        encode_features,
        metrics,
        predict,
        read_tsv,
        sha256_file,
        write_json,
    )

TEST_SOURCE_REVISION = "f578b5daac7bff3855d238c032a60fd047cddd9c"
TEST_SOURCE_PATH = "dataset/test_2.tsv"
TEST_SOURCE_GIT_BLOB = "de331fa36cbddfddbb178d7e3edda90139851a7a"
BASELINE_CANDIDATE = "line-distilbert-ja-first4"


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + bytes([0]) + data).hexdigest()


def audit_test_overlap(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> dict[str, int]:
    def pairs(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
        return {
            (normalize_sentence(row["premise"]), normalize_sentence(row["hypothesis"]))
            for row in rows
        }

    train_ids = {row["num"] for row in train_rows}
    validation_ids = {row["num"] for row in validation_rows}
    test_ids = {row["num"] for row in test_rows}
    train_groups = {row["template_num"] for row in train_rows}
    validation_groups = {row["template_num"] for row in validation_rows}
    test_groups = {row.get("template_num", "") for row in test_rows}
    train_pairs, validation_pairs, test_pairs = map(pairs, (train_rows, validation_rows, test_rows))
    overlap = {
        "train_source_id_overlap": len(train_ids & test_ids),
        "validation_source_id_overlap": len(validation_ids & test_ids),
        "train_template_group_overlap": len(train_groups & test_groups),
        "validation_template_group_overlap": len(validation_groups & test_groups),
        "train_normalized_pair_overlap": len(train_pairs & test_pairs),
        "validation_normalized_pair_overlap": len(validation_pairs & test_pairs),
        "test_duplicate_normalized_pairs": len(test_rows) - len(test_pairs),
    }
    if overlap["train_source_id_overlap"] or overlap["validation_source_id_overlap"]:
        raise ValueError(
            f"publisher test shares source IDs with frozen development data: {overlap}"
        )
    if overlap["train_normalized_pair_overlap"] or overlap["validation_normalized_pair_overlap"]:
        raise ValueError(
            f"publisher test shares normalized content with frozen development data: {overlap}"
        )
    return overlap


def validation_selection_record(
    frozen_report: dict[str, Any], adapted_report: dict[str, Any]
) -> dict[str, Any]:
    candidates = [
        {
            "candidate": item["candidate"],
            "validation_nll": item["selected_checkpoint"]["metrics"]["nll"],
            "selection": "minimum validation NLL within run",
        }
        for item in frozen_report["candidate_reports"]
    ]
    candidates.append(
        {
            "candidate": "line-distilbert-first4-top2-adapted",
            "validation_nll": adapted_report["best_checkpoint"]["metrics"]["nll"],
            "selection": "minimum validation NLL within run",
        }
    )
    selected = min(candidates, key=lambda item: item["validation_nll"])
    if selected["candidate"] != BASELINE_CANDIDATE:
        raise ValueError("predeclared validation-only selection rule changed before test access")
    return {
        "baseline": BASELINE_CANDIDATE,
        "rule": "minimum validation NLL across each run's validation-selected checkpoint",
        "candidate_validation_nll": candidates,
        "publisher_test_metrics_used_for_selection": False,
    }


def _prediction_metrics_by_category(rows: list[dict[str, str]], logits: Tensor) -> dict[str, Any]:
    result = {}
    targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in rows])
    for category in sorted({row["category"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["category"] == category]
        result[category] = metrics(logits[indices], targets[indices])
    return result


def _paired_group_bootstrap(
    rows: list[dict[str, str]],
    logits_by_candidate: dict[str, Tensor],
    *,
    baseline: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or any("template_num" not in row for row in rows):
        return {"status": "not_available", "reason": "test rows lack template_num groups"}
    groups = sorted({row["template_num"] for row in rows})
    if len(groups) < 5:
        return {
            "status": "not_available",
            "reason": "fewer than five independent template groups",
            "group_count": len(groups),
        }
    group_indices = {
        group: np.asarray([index for index, row in enumerate(rows) if row["template_num"] == group])
        for group in groups
    }
    targets = np.asarray([LABEL_TO_ID[row["gold_label"]] for row in rows])
    baseline_logits = logits_by_candidate[baseline].numpy()

    def per_row(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        prediction = probabilities.argmax(axis=1)
        accuracy = (prediction == targets).astype(np.float64)
        nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
        brier = ((probabilities - np.eye(len(LABELS))[targets]) ** 2).sum(axis=1)
        return accuracy, nll, brier

    base_metrics = per_row(baseline_logits)
    rng = np.random.Generator(np.random.PCG64(seed))
    output: dict[str, Any] = {
        "status": "computed",
        "baseline": baseline,
        "group_field": "template_num",
        "group_count": len(groups),
        "repetitions": repetitions,
        "seed": seed,
        "interval": "percentile 95% over paired template-group bootstrap",
        "comparisons": {},
    }
    for candidate, candidate_logits in logits_by_candidate.items():
        if candidate == baseline:
            continue
        candidate_metrics = per_row(candidate_logits.numpy())
        sampled_differences = {"accuracy": [], "nll": [], "brier": []}
        for _ in range(repetitions):
            selected_groups = rng.choice(groups, size=len(groups), replace=True)
            selected = np.concatenate([group_indices[group] for group in selected_groups])
            for key, candidate_values, base_values in zip(
                sampled_differences,
                candidate_metrics,
                base_metrics,
                strict=True,
            ):
                sampled_differences[key].append(
                    float(candidate_values[selected].mean() - base_values[selected].mean())
                )
        output["comparisons"][candidate] = {
            key: {
                "mean_difference": float(np.mean(values)),
                "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
            for key, values in sampled_differences.items()
        }
    return output


def _load_selected_frozen_candidate(
    *,
    name: str,
    model_dir: Path,
    line_layers: int,
    revision: str,
    run_dir: Path,
    test_rows: list[dict[str, str]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], Tensor]:
    report_path = run_dir / "comparison-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("official_test_loaded", False):
        raise ValueError("training run already loaded an official publisher test")
    candidate_report = next(
        item for item in report["candidate_reports"] if item["candidate"] == name
    )
    checkpoint_path = run_dir / name / "best-readout.pt"
    if sha256_file(checkpoint_path) != candidate_report["best_readout_sha256"]:
        raise ValueError(f"selected readout hash mismatch for {name}")
    feature_path = output_dir / f"{name}-features.npy"
    encoder_metadata = encode_features(
        model_dir,
        test_rows,
        feature_path,
        candidate=name,
        line_layers=line_layers,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    features = np.load(feature_path, mmap_mode="r")
    scorer = Readout(int(features.shape[-1]))
    scorer.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    scorer.to(device).eval()
    targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in test_rows])
    logits, summary = predict(scorer, features, targets, device, 512)
    prediction_path = output_dir / f"{name}-test-predictions.jsonl"
    _save_predictions(prediction_path, test_rows, logits)
    candidate_result = {
        "candidate": name,
        "model_revision": revision,
        "validation_selected_epoch": candidate_report["selected_checkpoint"]["epoch"],
        "validation_selected_metrics": candidate_report["selected_checkpoint"]["metrics"],
        "test_metrics": summary,
        "test_metrics_by_category": _prediction_metrics_by_category(test_rows, logits),
        "test_metrics_by_template": _prediction_metrics_by_category(
            [dict(row, category=row["template_num"]) for row in test_rows], logits
        )
        if test_rows and "template_num" in test_rows[0]
        else {},
        "encoder_parameters": encoder_metadata["encoder_parameters"],
        "readout_parameters": candidate_report["readout_trainable_parameters"],
        "total_text_parameters": candidate_report["total_text_parameters"],
        "four_tower_parameters_if_replacing_ruri": candidate_report[
            "four_tower_parameters_if_replacing_ruri"
        ],
        "test_encoder": encoder_metadata,
        "readout_sha256": candidate_report["best_readout_sha256"],
        "test_prediction_sha256": sha256_file(prediction_path),
    }
    return candidate_result, logits


def _load_selected_adapted_candidate(
    *,
    model_dir: Path,
    run_dir: Path,
    test_rows: list[dict[str, str]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], Tensor]:
    run_report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best-checkpoint.pt"
    if sha256_file(checkpoint_path) != run_report["best_checkpoint_sha256"]:
        raise ValueError("adapted DistilBERT selected checkpoint hash mismatch")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "distilbert" or int(config.n_layers) != 6:
        raise ValueError("unexpected base DistilBERT configuration")
    config.n_layers = 4
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True,
    )
    encoder = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    readout = Readout(int(config.hidden_size))
    readout.load_state_dict(checkpoint["readout"])
    readout.to(device).eval()
    tokens, token_stats = _tokenize(tokenizer, test_rows, max_length=max_length)
    targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in test_rows])
    logits = _encode_logits(
        encoder,
        readout,
        tokens,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    summary = metrics(logits, targets)
    prediction_path = output_dir / "line-distilbert-first4-top2-test-predictions.jsonl"
    _save_predictions(prediction_path, test_rows, logits)
    candidate_result = {
        "candidate": "line-distilbert-first4-top2-adapted",
        "model_revision": run_report["model"]["revision"],
        "validation_selected_epoch": run_report["best_checkpoint"]["epoch"],
        "validation_selected_metrics": run_report["best_checkpoint"]["metrics"],
        "test_metrics": summary,
        "test_metrics_by_category": _prediction_metrics_by_category(test_rows, logits),
        "test_metrics_by_template": _prediction_metrics_by_category(
            [dict(row, category=row["template_num"]) for row in test_rows], logits
        )
        if test_rows and "template_num" in test_rows[0]
        else {},
        "encoder_parameters": run_report["model"]["total_text_parameters"]
        - run_report["model"]["readout_trainable_parameters"],
        "readout_parameters": run_report["model"]["readout_trainable_parameters"],
        "total_text_parameters": run_report["model"]["total_text_parameters"],
        "four_tower_parameters_if_replacing_ruri": run_report["model"][
            "four_tower_parameters_if_replacing_ruri"
        ],
        "test_truncation": token_stats,
        "checkpoint_sha256": run_report["best_checkpoint_sha256"],
        "test_prediction_sha256": sha256_file(prediction_path),
    }
    return candidate_result, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-tsv", type=Path, required=True)
    parser.add_argument("--expected-test-git-blob", default=TEST_SOURCE_GIT_BLOB)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-run-dir", type=Path, required=True)
    parser.add_argument("--adapted-run-dir", type=Path, required=True)
    parser.add_argument("--ruri-model-dir", type=Path, required=True)
    parser.add_argument("--line-distilbert-dir", type=Path, required=True)
    parser.add_argument("--tohoku-bert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a prior publisher-test evaluation")
    train_rows = read_tsv(args.data_dir / "train.tsv")
    validation_rows = read_tsv(args.data_dir / "validation.tsv")
    frozen_run = json.loads(
        (args.frozen_run_dir / "comparison-report.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads((args.data_dir / "split-manifest.json").read_text(encoding="utf-8"))
    expected_split_hashes = {
        "train": split_manifest["train_sha256"],
        "validation": split_manifest["validation_sha256"],
    }
    if frozen_run["split_manifest"]["frozen_train_sha256"] != expected_split_hashes["train"]:
        raise ValueError("training comparison uses a different frozen training split")
    if (
        frozen_run["split_manifest"]["frozen_validation_sha256"]
        != expected_split_hashes["validation"]
    ):
        raise ValueError("training comparison uses a different frozen validation split")
    adapted_report = json.loads(
        (args.adapted_run_dir / "run-report.json").read_text(encoding="utf-8")
    )
    if adapted_report["train_sha256"] != expected_split_hashes["train"]:
        raise ValueError("adapted run uses a different frozen training split")
    if adapted_report["validation_sha256"] != expected_split_hashes["validation"]:
        raise ValueError("adapted run uses a different frozen validation split")
    if frozen_run.get("official_test_loaded", True):
        raise ValueError("frozen training run did not certify publisher test isolation")
    if adapted_report.get("official_test_loaded", True):
        raise ValueError("adapted training run did not certify publisher test isolation")
    selection = validation_selection_record(frozen_run, adapted_report)

    actual_blob = git_blob_sha1(args.test_tsv)
    if actual_blob != args.expected_test_git_blob:
        raise ValueError(f"publisher test git blob mismatch: {actual_blob}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    test_rows = read_tsv(args.test_tsv)
    if not test_rows:
        raise ValueError("publisher test has no rows")
    if any(row["gold_label"] not in LABELS for row in test_rows):
        raise ValueError("publisher test contains unknown labels")
    if len({row["num"] for row in test_rows}) != len(test_rows):
        raise ValueError("publisher test contains duplicate source IDs")
    overlap = audit_test_overlap(train_rows, validation_rows, test_rows)

    specs = [
        ("ruri-v3-30m", args.ruri_model_dir, 0, "24899e5de370b56d179604a007c0d727bf144504"),
        (
            "line-distilbert-ja-first4",
            args.line_distilbert_dir,
            4,
            "c32d05753d0d4be0f60482291a9452ec7e6f7d68",
        ),
        (
            "tohoku-bert-base-ja-v3",
            args.tohoku_bert_dir,
            0,
            "65243d6e5629b969c77309f217bd7b1a79d43c7e",
        ),
    ]
    results, logits_by_candidate = [], {}
    for name, model_dir, layers, revision in specs:
        result, logits = _load_selected_frozen_candidate(
            name=name,
            model_dir=model_dir,
            line_layers=layers,
            revision=revision,
            run_dir=args.frozen_run_dir,
            test_rows=test_rows,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        results.append(result)
        logits_by_candidate[name] = logits
    adapted_result, adapted_logits = _load_selected_adapted_candidate(
        model_dir=args.line_distilbert_dir,
        run_dir=args.adapted_run_dir,
        test_rows=test_rows,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    results.append(adapted_result)
    logits_by_candidate[adapted_result["candidate"]] = adapted_logits

    result = {
        "experiment_id": "jamp-publisher-test-one-shot-bert-comparison-v1",
        "source_revision": TEST_SOURCE_REVISION,
        "source_path": TEST_SOURCE_PATH,
        "source_declared_license": "CC-BY-SA-4.0",
        "test_file_git_blob_sha1": actual_blob,
        "test_file_sha256": sha256_file(args.test_tsv),
        "test_file_bytes": args.test_tsv.stat().st_size,
        "test_examples": len(test_rows),
        "test_label_counts": {
            label: sum(row["gold_label"] == label for row in test_rows) for label in LABELS
        },
        "train_validation_test_overlap_audit": overlap,
        "selection": selection,
        "baseline": BASELINE_CANDIDATE,
        "paired_template_group_bootstrap": _paired_group_bootstrap(
            test_rows,
            logits_by_candidate,
            baseline=BASELINE_CANDIDATE,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        ),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "candidates": results,
        "official_test_loaded_once": True,
        "test_reuse_prohibited": True,
        "test_metrics_used_for_selection": False,
        "product_promotion": False,
    }
    write_json(args.output_dir / "one-shot-test-result.json", result)
    print(
        json.dumps(
            {
                "test_examples": len(test_rows),
                "test_label_counts": result["test_label_counts"],
                "overlap": overlap,
                "candidates": [
                    {
                        "candidate": item["candidate"],
                        "test_metrics": item["test_metrics"],
                        "params": item["total_text_parameters"],
                        "four_tower_params": item["four_tower_parameters_if_replacing_ruri"],
                    }
                    for item in results
                ],
                "bootstrap": result["paired_template_group_bootstrap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
