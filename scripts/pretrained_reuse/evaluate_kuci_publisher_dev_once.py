"""One-shot transfer evaluation on the untouched official KUCI development split.

This script never fits or selects a checkpoint. It requires the previously
selected readouts and refuses to overwrite an output directory. The dataset is
public development data, not a sealed audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

try:
    from scripts.analyze_kuci_temporal_probe import paired_group_bootstrap
    from scripts.build_kuci_train_grouped_split import sha256_file
    from scripts.train_kuci_frozen_text_readout import (
        CandidateScorer,
        _assert_option_order_equivariance,
        _cpu_state_dict,
        _metrics,
        _predict,
        _save_predictions,
        encode_features,
    )
except ModuleNotFoundError:
    from analyze_kuci_temporal_probe import paired_group_bootstrap
    from build_kuci_train_grouped_split import sha256_file
    from train_kuci_frozen_text_readout import (
        CandidateScorer,
        _assert_option_order_equivariance,
        _cpu_state_dict,
        _metrics,
        _predict,
        _save_predictions,
        encode_features,
    )

SOURCE_REVISION = "79afa1d63d1f262e6709e9be714d9ad5dae8d6f6"
SOURCE_URL = (
    "https://github.com/ku-nlp/KUCI/blob/"
    f"{SOURCE_REVISION}/dataset/dev.jsonl"
)
LABELS = set("abcd")
EXPECTED_READOUTS = {
    "ruri-v3-30m": "43d4e19f2bed1645c1e67d87af99d24b82e1633d949e485fa92999029580f3e8",
    "line-distilbert-ja-first4": (
        "3164b38c54bd18b73ebddbf5c2b0edde379f7b700dbc9cf376b8a660cc28ec2a"
    ),
}
REQUIRED = {
    "id",
    "context",
    "core_event_pair",
    "label",
    "agreement",
    *(f"choice_{letter}" for letter in "abcd"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def content_key(row: dict[str, Any]) -> str:
    options = sorted(normalize_text(str(row[f"choice_{letter}"])) for letter in "abcd")
    value = "\0".join((normalize_text(str(row["context"])), *options))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_rows(
    rows: list[dict[str, Any]], references: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("publisher development split is empty")
    ids: list[str] = []
    contents: list[str] = []
    contexts: list[str] = []
    event_pairs: list[str] = []
    for row in rows:
        if REQUIRED - row.keys():
            raise ValueError("KUCI row is missing required fields")
        if str(row["label"]) not in LABELS:
            raise ValueError("KUCI target must be one of a/b/c/d")
        if len({normalize_text(str(row[f"choice_{letter}"])) for letter in "abcd"}) != 4:
            raise ValueError("every KUCI row must contain four distinct normalized choices")
        ids.append(str(row["id"]))
        contents.append(content_key(row))
        contexts.append(normalize_text(str(row["context"])))
        event_pairs.append(normalize_text(str(row["core_event_pair"])))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate publisher development IDs")
    if len(contents) != len(set(contents)):
        raise ValueError(
            "duplicate normalized question/choice content within publisher development"
        )

    summary: dict[str, Any] = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_content": len(set(contents)),
        "label_counts": {
            label: sum(str(row["label"]) == label for row in rows) for label in "abcd"
        },
        "distinct_event_pairs": len(set(event_pairs)),
        "references": {},
    }
    dev_id_set, content_set = set(ids), set(contents)
    context_set, event_pair_set = set(contexts), set(event_pairs)
    for name, other_rows in references.items():
        reference_ids = {str(row["id"]) for row in other_rows}
        reference_content = {content_key(row) for row in other_rows}
        reference_context = {normalize_text(str(row["context"])) for row in other_rows}
        reference_event_pairs = {
            normalize_text(str(row["core_event_pair"])) for row in other_rows
        }
        summary["references"][name] = {
            "rows": len(other_rows),
            "shared_ids": len(dev_id_set & reference_ids),
            "shared_normalized_content": len(content_set & reference_content),
            "shared_normalized_contexts": len(context_set & reference_context),
            "shared_event_pairs": len(event_pair_set & reference_event_pairs),
        }
    return summary


def assert_no_training_group_leakage(
    audit: dict[str, Any], reference: str = "official_train"
) -> None:
    overlap = audit["references"][reference]
    if overlap["shared_normalized_content"]:
        raise ValueError("publisher development content overlaps official training data")
    if overlap["shared_event_pairs"]:
        raise ValueError("publisher development event groups overlap official training data")


def _targets(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor(["abcd".index(str(row["label"])) for row in rows], dtype=torch.long)


def _load_selected_readout(path: Path, hidden_size: int, device: torch.device) -> CandidateScorer:
    scorer = CandidateScorer(hidden_size)
    state = torch.load(path, map_location="cpu", weights_only=True)
    scorer.load_state_dict(state, strict=True)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    scorer.to(device).eval()
    if any(parameter.requires_grad for parameter in scorer.parameters()):
        raise ValueError("evaluation readout unexpectedly has trainable parameters")
    return scorer


def evaluate_candidate(
    *,
    name: str,
    model_dir: Path,
    line_layers: int,
    revision: str,
    readout_path: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_questions: int,
) -> tuple[dict[str, Any], Path]:
    output_dir.mkdir(parents=False, exist_ok=False)
    features_path = output_dir / "features.npy"
    started = time.perf_counter()
    encoder = encode_features(
        model_dir,
        rows,
        features_path,
        device=device,
        line_layers=line_layers,
        batch_questions=batch_questions,
        max_length=512,
    )
    features = np.load(features_path, mmap_mode="r")
    scorer = _load_selected_readout(readout_path, int(features.shape[-1]), device)
    logits, metrics = _predict(scorer, features, _targets(rows), device)
    option_order_delta = _assert_option_order_equivariance(scorer, features, device)
    prediction_path = output_dir / "publisher-dev-predictions.jsonl"
    _save_predictions(prediction_path, rows, logits)
    result = {
        "candidate": name,
        "model_revision": revision,
        "model_files_sha256": encoder["weight_files_sha256"],
        "encoder_parameters": encoder["encoder_parameters"],
        "trainable_encoder_parameters": encoder["trainable_encoder_parameters"],
        "readout_sha256": sha256_file(readout_path),
        "readout_parameters": sum(value.numel() for value in _cpu_state_dict(scorer).values()),
        "metrics": metrics,
        "agreement_metrics": {
            str(agreement): _metrics(logits[indices], _targets([rows[index] for index in indices]))
            for agreement in sorted({int(row["agreement"]) for row in rows})
            for indices in [[i for i, row in enumerate(rows) if int(row["agreement"]) == agreement]]
        },
        "option_order_equivariance_max_abs_delta": option_order_delta,
        "encoder": encoder,
        "evaluation_seconds": time.perf_counter() - started,
        "prediction_sha256": sha256_file(prediction_path),
    }
    del scorer, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-dev", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--frozen-corpus", type=Path, required=True)
    parser.add_argument("--line-model", type=Path, required=True)
    parser.add_argument("--ruri-model", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-questions", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite publisher evaluation: {args.output}")
    if args.batch_questions <= 0:
        raise ValueError("batch-questions must be positive")
    for path in (args.publisher_dev, args.official_train):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.frozen_corpus.is_dir():
        raise FileNotFoundError(args.frozen_corpus)
    for model_dir in (args.line_model, args.ruri_model):
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)

    rows = read_jsonl(args.publisher_dev)
    official_train = read_jsonl(args.official_train)
    frozen_train = read_jsonl(args.frozen_corpus / "train.jsonl")
    frozen_validation = read_jsonl(args.frozen_corpus / "validation.jsonl")
    audit = audit_rows(
        rows,
        {
            "official_train": official_train,
            "frozen_train": frozen_train,
            "frozen_validation": frozen_validation,
        },
    )
    if args.audit_only:
        print(
            json.dumps(
                {
                    "publisher_dev_sha256": sha256_file(args.publisher_dev),
                    "official_train_sha256": sha256_file(args.official_train),
                    "frozen_train_sha256": sha256_file(args.frozen_corpus / "train.jsonl"),
                    "frozen_validation_sha256": sha256_file(
                        args.frozen_corpus / "validation.jsonl"
                    ),
                    "audit": audit,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    assert_no_training_group_leakage(audit)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    selected = {
        "ruri-v3-30m": {
            "model_dir": args.ruri_model,
            "line_layers": 0,
            "revision": "24899e5de370b56d179604a007c0d727bf144504",
            "readout": args.comparison_dir / "ruri-v3-30m" / "best-readout.pt",
        },
        "line-distilbert-ja-first4": {
            "model_dir": args.line_model,
            "line_layers": 4,
            "revision": "c32d05753d0d4be0f60482291a9452ec7e6f7d68",
            "readout": args.comparison_dir / "line-distilbert-ja-first4" / "best-readout.pt",
        },
    }
    if any(not value["readout"].is_file() for value in selected.values()):
        raise FileNotFoundError("a frozen validation-selected readout is missing")
    for name, item in selected.items():
        if sha256_file(item["readout"]) != EXPECTED_READOUTS[name]:
            raise ValueError(f"frozen selected readout hash mismatch: {name}")

    args.output.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    prediction_paths: dict[str, Path] = {}
    for name, item in selected.items():
        result, prediction_path = evaluate_candidate(
            name=name,
            model_dir=item["model_dir"],
            line_layers=item["line_layers"],
            revision=item["revision"],
            readout_path=item["readout"],
            rows=rows,
            output_dir=args.output / name,
            device=device,
            batch_questions=args.batch_questions,
        )
        results[name] = result
        prediction_paths[name] = prediction_path

    bootstrap = paired_group_bootstrap(
        rows,
        read_jsonl(prediction_paths["ruri-v3-30m"]),
        read_jsonl(prediction_paths["line-distilbert-ja-first4"]),
        replicates=10_000,
        seed=1_729,
    )
    bootstrap["resampling_unit"] = (
        "connected publisher-development component (event pair, normalized context, option content)"
    )
    bootstrap["evaluation_rows"] = bootstrap.pop("validation_rows")
    bootstrap["limitations"] = [
        "checkpoints were selected on a separate internal grouped validation split",
        "this is public publisher development data, not a sealed audit",
        "normalized contexts overlap official train (976 rows) and internal validation (82 rows)",
        "the intervals are descriptive and do not establish broad Japanese or multimodal quality",
    ]
    report = {
        "schema_version": 1,
        "evaluation_id": "kuci-official-dev-one-shot-20261124",
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "publisher_dev_sha256": sha256_file(args.publisher_dev),
        "official_train_sha256": sha256_file(args.official_train),
        "frozen_train_sha256": sha256_file(args.frozen_corpus / "train.jsonl"),
        "frozen_validation_sha256": sha256_file(args.frozen_corpus / "validation.jsonl"),
        "frozen_split_audit": audit,
        "evaluation_policy": {
            "one_shot": True,
            "checkpoint_selection": "previous internal grouped validation NLL only",
            "training_or_calibration_on_publisher_dev": False,
            "candidate_set": list(selected),
            "sealed_audit": False,
        },
        "models": results,
        "paired_group_bootstrap": bootstrap,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        },
        "output_predictions_sha256": {
            name: sha256_file(path) for name, path in prediction_paths.items()
        },
    }
    report_path = args.output / "evaluation-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
