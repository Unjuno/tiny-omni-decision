"""Compare frozen Japanese BERT encoders as a binary option Decision model.

Only JCoLA's in-domain training data and its source-independent out-of-domain
development split are used. The in-domain public validation file is not read:
its source groups overlap all in-domain training sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

OPTIONS = ("不適格（文法的に許容されない）", "適格（文法的に許容される）")
QUESTION = "次の文について、文法上の許容性を選んでください。"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def split_by_source(
    rows: list[dict[str, Any]], seed: int, validation_source_count: int = 2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    sources = sorted({str(row["source"]) for row in rows})
    if len(sources) <= validation_source_count:
        raise ValueError("not enough source groups for a train/validation split")
    random.Random(seed).shuffle(sources)
    validation_sources = sorted(sources[:validation_source_count])
    train_sources = sorted(sources[validation_source_count:])
    train = [row for row in rows if str(row["source"]) in train_sources]
    validation = [row for row in rows if str(row["source"]) in validation_sources]
    if not train or not validation:
        raise ValueError("source split produced an empty partition")
    return train, validation, train_sources, validation_sources


def validate_corpus(rows: list[dict[str, Any]], name: str) -> None:
    ids = [str(row["uid"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} has duplicate uids")
    for row in rows:
        if row.get("label") not in (0, 1):
            raise ValueError(f"{name} has a non-binary label")
        if not str(row.get("sentence", "")).strip() or not str(row.get("source", "")):
            raise ValueError(f"{name} is missing sentence/source")


def assert_disjoint_groups_and_content(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], name: str
) -> None:
    train_sources = {str(row["source"]) for row in train}
    validation_sources = {str(row["source"]) for row in validation}
    if train_sources & validation_sources:
        raise ValueError(f"{name} source groups cross partitions")
    train_text = {normalize_sentence(str(row["sentence"])) for row in train}
    validation_text = {normalize_sentence(str(row["sentence"])) for row in validation}
    if train_text & validation_text:
        raise ValueError(f"{name} normalized sentence content crosses partitions")


class CandidateScorer(nn.Module):
    def __init__(self, hidden_size: int, readout_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, readout_size),
            nn.Tanh(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)


def decision_pairs(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    first: list[str] = []
    second: list[str] = []
    for row in rows:
        context = f"{QUESTION}\n文: {row['sentence']}"
        for option in OPTIONS:
            first.append(context)
            second.append(option)
    return first, second


def encode_rows(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    line_distilbert_layers: int = 0,
) -> tuple[Tensor, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if line_distilbert_layers:
        if config.model_type != "distilbert" or line_distilbert_layers >= int(config.n_layers):
            raise ValueError("layer truncation is restricted to a shallower DistilBERT")
        config.n_layers = line_distilbert_layers
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=(config.model_type == "distilbert"),
    )
    try:
        model, loading_info = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            add_pooling_layer=False,
            output_loading_info=True,
        )
    except TypeError as error:
        if "unexpected keyword argument 'add_pooling_layer'" not in str(error):
            raise
        model, loading_info = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            output_loading_info=True,
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first, second = decision_pairs(rows)
    chunks: list[Tensor] = []
    max_tokens = 0
    truncated = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(first), batch_size):
            end = start + batch_size
            raw = tokenizer(first[start:end], second[start:end], truncation=False, padding=False)
            lengths = [len(item) for item in raw["input_ids"]]
            max_tokens = max(max_tokens, max(lengths, default=0))
            truncated += sum(length > max_length for length in lengths)
            batch = tokenizer(
                first[start:end],
                second[start:end],
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            embedding_rows = model.get_input_embeddings().num_embeddings
            if int(batch["input_ids"].max()) >= embedding_rows:
                raise ValueError("tokenizer emitted an ID outside the encoder embedding table")
            output = model(**batch).last_hidden_state[:, 0].float().cpu()
            if not torch.isfinite(output).all():
                raise ValueError("encoder produced non-finite CLS features")
            chunks.append(output)
            del output, batch, raw
    elapsed = time.perf_counter() - started
    features = torch.cat(chunks).reshape(len(rows), 2, -1).contiguous()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()
    ):
        raise ValueError("frozen encoder has trainable parameters or gradients")
    if loading_info["missing_keys"]:
        raise ValueError(
            f"checkpoint did not initialize every encoder parameter: {loading_info['missing_keys']}"
        )
    if line_distilbert_layers:
        unexpected = [str(key) for key in loading_info["unexpected_keys"]]
        dropped_indices: set[int] = set()
        for key in unexpected:
            match = re.search(r"transformer\.layer\.\{?([0-9, ]+)\}?\.", key)
            if match:
                dropped_indices.update(int(value) for value in re.findall(r"\d+", match.group(1)))
        if dropped_indices != {4, 5}:
            raise ValueError(
                "4-layer DistilBERT did not load exactly the first four pretrained layers"
            )
    meta = {
        "model_type": config.model_type,
        "model_revision": getattr(config, "_commit_hash", None),
        "layers": int(getattr(config, "n_layers", getattr(config, "num_hidden_layers", 0))),
        "hidden_size": int(config.hidden_size),
        "encoder_parameters": parameters,
        "trainable_encoder_parameters": 0,
        "embedding_rows": model.get_input_embeddings().num_embeddings,
        "tokenizer_vocabulary_size": int(tokenizer.vocab_size),
        "missing_keys": sorted(str(key) for key in loading_info["missing_keys"]),
        "unexpected_keys": sorted(str(key) for key in loading_info["unexpected_keys"]),
        "options_per_decision": 2,
        "encoded_candidate_pairs": len(first),
        "max_untruncated_tokens": max_tokens,
        "truncated_candidate_pairs": truncated,
        "feature_extraction_seconds": elapsed,
        "frozen": True,
        "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, meta


def metrics(logits: Tensor, targets: Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(-1)
    targets = targets.long()
    correct = predictions.eq(targets)
    recalls = [float(correct[targets == label].float().mean()) for label in (0, 1)]
    confidence = probabilities.max(-1).values
    ece = 0.0
    for index in range(15):
        lo, hi = index / 15, (index + 1) / 15
        selected = (confidence >= lo) & ((confidence < hi) if index < 14 else (confidence <= hi))
        if selected.any():
            ece += float(selected.float().mean()) * abs(
                float(confidence[selected].mean()) - float(correct[selected].float().mean())
            )
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": sum(recalls) / 2,
        "unacceptable_recall": recalls[0],
        "acceptable_recall": recalls[1],
        "nll": float(F.cross_entropy(logits, targets)),
        "brier": float((probabilities - F.one_hot(targets, 2)).square().sum(-1).mean()),
        "ece_15": ece,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "mean_confidence": float(confidence.mean()),
    }


def fit_readout(
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[CandidateScorer, dict[str, Any]]:
    torch.manual_seed(seed)
    scorer = CandidateScorer(int(train_x.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    best_nll, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history: list[dict[str, float | int]] = []
    train_y = train_y.to(device)
    validation_y = validation_y.to(device)
    train_x = train_x.to(device)
    validation_x = validation_x.to(device)
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_x)))
        rng.shuffle(order)
        train_losses = []
        scorer.train()
        for start in range(0, len(order), batch_size):
            indices = torch.tensor(order[start : start + batch_size], device=device)
            logits = scorer(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            validation_logits = scorer(validation_x)
            validation_metrics = metrics(validation_logits.cpu(), validation_y.cpu())
        validation_nll = float(validation_metrics["nll"])
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(train_losses) / len(train_losses),
                "selection_validation_nll": validation_nll,
                "selection_validation_accuracy": float(validation_metrics["accuracy"]),
                "selection_validation_macro_accuracy": float(validation_metrics["macro_accuracy"]),
            }
        )
        if validation_nll < best_nll - 1e-10:
            best_nll, best_epoch = validation_nll, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    scorer.load_state_dict(best_state)
    scorer.to(device).eval()
    return scorer, {"best_epoch": best_epoch, "epochs_completed": len(history), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bert-base-dir", type=Path, required=True)
    parser.add_argument("--line-distilbert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    data_dir = args.source_dir / "data" / "jcola-v1.0"
    source_revision = subprocess.run(
        ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != "736d9eef3af04bb17e4cf59adf01325973234ff9":
        raise ValueError(f"unexpected JCoLA source revision: {source_revision}")
    train_path = data_dir / "in_domain_train-v1.0.json"
    ood_path = data_dir / "out_of_domain_valid-v1.0.json"
    assignment_path = args.source_dir / "source-grouped-train-split-assignments.jsonl"
    split_manifest_path = args.source_dir / "jcola-source-grouped-split-manifest.json"
    if not assignment_path.is_file() or not split_manifest_path.is_file():
        raise FileNotFoundError("frozen source-grouped split manifest/assignments are required")
    train_rows = read_jsonl(train_path)
    validate_corpus(train_rows, "in-domain train")
    train, selection_validation, train_sources, validation_sources = split_by_source(
        train_rows, args.seed
    )
    frozen_assignments = {item["uid"]: item["split"] for item in read_jsonl(assignment_path)}
    expected_assignments = {
        str(row["uid"]): "validation" if row["source"] in validation_sources else "train"
        for row in train_rows
    }
    if frozen_assignments != expected_assignments:
        raise ValueError("recomputed source split differs from the frozen assignments")
    assert_disjoint_groups_and_content(train, selection_validation, "in-domain train split")
    if args.smoke_rows:
        train = train[: args.smoke_rows]
        selection_validation = selection_validation[: max(1, args.smoke_rows // 4)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_started = time.time()
    models = [
        ("tohoku-bert-base-v3", args.bert_base_dir, 0),
        ("line-distilbert-ja-4layer", args.line_distilbert_dir, 4),
    ]
    reports = []
    for name, model_dir, layer_count in models:
        model_output = args.output_dir / name
        model_output.mkdir()
        combined_x, train_meta = encode_rows(
            model_dir,
            train + selection_validation,
            device=device,
            batch_size=args.encode_batch_size,
            max_length=args.max_length,
            line_distilbert_layers=layer_count,
        )
        train_x = combined_x[: len(train)]
        selection_x = combined_x[len(train) :]
        train_y = torch.tensor([int(row["label"]) for row in train], dtype=torch.long)
        selection_y = torch.tensor(
            [int(row["label"]) for row in selection_validation], dtype=torch.long
        )
        scorer, fit_meta = fit_readout(
            train_x,
            train_y,
            selection_x,
            selection_y,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=device,
        )
        torch.save(scorer.cpu().state_dict(), model_output / "best-readout.pt")
        report = {
            "model": name,
            "selection_validation_source_groups": validation_sources,
            "training_source_groups": train_sources,
            "training_examples": len(train),
            "selection_validation_examples": len(selection_validation),
            "selection_validation_labels": dict(
                Counter(str(int(r["label"])) for r in selection_validation)
            ),
            "encoder": train_meta,
            "model_weight_sha256": {
                path.name: sha256_file(path)
                for path in model_dir.iterdir()
                if path.is_file() and path.name in {"model.safetensors", "pytorch_model.bin"}
            },
            "trainable_readout_parameters": sum(p.numel() for p in scorer.parameters()),
            "total_parameters_before_quantization": train_meta["encoder_parameters"]
            + sum(p.numel() for p in scorer.parameters()),
            "selection": fit_meta,
            "best_readout_sha256": sha256_file(model_output / "best-readout.pt"),
        }
        reports.append(report)
        del scorer, combined_x, train_x, selection_x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.smoke_rows:
        smoke_manifest = {
            "smoke_only": True,
            "seed": args.seed,
            "source_revision": source_revision,
            "train_sha256": sha256_file(train_path),
            "split_assignments_sha256": sha256_file(assignment_path),
            "in_domain_publisher_validation_loaded": False,
            "official_out_of_domain_development_loaded_or_scored": False,
            "official_test_or_sealed_audit_loaded": False,
            "models": reports,
        }
        (args.output_dir / "smoke-manifest.json").write_text(
            json.dumps(smoke_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"smoke_only": True, "models": [r["model"] for r in reports]}))
        return
    # Model choices and checkpoints are frozen before any out-of-domain
    # development examples are passed through either selected model.
    ood_rows = read_jsonl(ood_path)
    validate_corpus(ood_rows, "official out-of-domain development")
    assert_disjoint_groups_and_content(train_rows, ood_rows, "train/out-of-domain development")
    for report, (name, model_dir, layer_count) in zip(reports, models, strict=True):
        model_output = args.output_dir / name
        ood_x, ood_meta = encode_rows(
            model_dir,
            ood_rows,
            device=device,
            batch_size=args.encode_batch_size,
            max_length=args.max_length,
            line_distilbert_layers=layer_count,
        )
        scorer = CandidateScorer(int(ood_x.shape[-1])).to(device)
        scorer.load_state_dict(
            torch.load(model_output / "best-readout.pt", map_location=device, weights_only=True)
        )
        scorer.eval()
        ood_y = torch.tensor([int(row["label"]) for row in ood_rows], dtype=torch.long)
        with torch.inference_mode():
            ood_logits = scorer(ood_x.to(device)).cpu()
            swapped_logits = scorer(ood_x[:, [1, 0]].to(device)).cpu()
        swap_delta = float((swapped_logits - ood_logits[:, [1, 0]]).abs().max())
        if swap_delta > 1e-7:
            raise ValueError("swapping supplied options changed more than score order")
        report["out_of_domain_encoder"] = ood_meta
        report["out_of_domain_development_metrics"] = metrics(ood_logits, ood_y)
        report["option_swap_max_score_delta"] = swap_delta
        predictions = [
            {
                "uid": str(row["uid"]),
                "source": row["source"],
                "target": int(row["label"]),
                "prediction": int(ood_logits[i].argmax()),
                "p_unacceptable": float(torch.softmax(ood_logits[i], -1)[0]),
                "p_acceptable": float(torch.softmax(ood_logits[i], -1)[1]),
            }
            for i, row in enumerate(ood_rows)
        ]
        (model_output / "out-of-domain-development-predictions.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions
            ),
            encoding="utf-8",
        )
        (model_output / "run-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        del scorer, ood_x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    effective_config = {
        "seed": args.seed,
        "models": [
            {"name": name, "path": str(path.resolve()), "distilbert_layer_limit": layers}
            for name, path, layers in models
        ],
        "readout": "frozen CLS -> 128 tanh -> one shared option score",
        "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size_decisions": args.batch_size,
        "encode_batch_size_candidate_pairs": args.encode_batch_size,
        "max_length": args.max_length,
        "source_revision": source_revision,
        "train_sha256": sha256_file(train_path),
        "official_out_of_domain_development_sha256": sha256_file(ood_path),
        "split_assignments_sha256": sha256_file(assignment_path),
        "selection_validation_source_groups": validation_sources,
        "official_out_of_domain_evaluation_after_checkpoint_selection": True,
    }
    effective_config_bytes = json.dumps(
        effective_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run = {
        "task": "JCoLA binary supplied-option linguistic acceptability Decision",
        "seed": args.seed,
        "device": str(device),
        "source_revision": source_revision,
        "source_license": "CC-BY-SA-4.0; artifacts remain local and data is not copied to Git",
        "train_sha256": sha256_file(train_path),
        "official_out_of_domain_development_sha256": sha256_file(ood_path),
        "split_assignments_sha256": sha256_file(assignment_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "grouping_and_overlap": {
            "train_source_groups": len({row["source"] for row in train}),
            "selection_validation_source_groups": len(
                {row["source"] for row in selection_validation}
            ),
            "train_selection_source_overlap": 0,
            "train_selection_normalized_content_overlap": 0,
            "official_ood_source_groups": len({row["source"] for row in ood_rows}),
            "train_ood_source_overlap": 0,
            "train_ood_normalized_content_overlap": 0,
        },
        "in_domain_publisher_validation_loaded": False,
        "official_test_or_sealed_audit_loaded": False,
        "in_domain_validation_exclusion_reason": "All 10 source groups overlap in-domain train.",
        "inner_validation_selection_source_groups": validation_sources,
        "output_order": list(OPTIONS),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
        },
        "hyperparameters": {
            "readout": "frozen CLS -> 128 tanh -> one shared option score",
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size_decisions": args.batch_size,
            "encode_batch_size_candidate_pairs": args.encode_batch_size,
            "max_length": args.max_length,
        },
        "effective_config": effective_config,
        "effective_config_sha256": hashlib.sha256(effective_config_bytes).hexdigest(),
        "training_script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - run_started,
        "models": reports,
    }
    (args.output_dir / "run-manifest.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "models": [
                    {
                        "model": r["model"],
                        "best_epoch": r["selection"]["best_epoch"],
                        "ood_metrics": r["out_of_domain_development_metrics"],
                    }
                    for r in reports
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
