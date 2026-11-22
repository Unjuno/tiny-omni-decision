"""Compare frozen and upper-layer-adapted compact Japanese BERT on grouped CV.
        "adapted_optimizer": (
            f"AdamW lr={args.learning_rate}, wd={args.weight_decay}; grad clip 1.0"
        ),

This experiment uses only the pinned JCoLA in-domain TRAIN file. It deliberately
does not open either publisher validation file, the public OOD development set,
or publisher test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from compare_jcola_bert_decision import (
    OPTIONS,
    QUESTION,
    CandidateScorer,
    assert_disjoint_groups_and_content,
    metrics,
    normalize_sentence,
    read_jsonl,
    sha256_file,
    validate_corpus,
)
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

SOURCE_REVISION = "736d9eef3af04bb17e4cf59adf01325973234ff9"
MODEL_REVISION = "c32d05753d0d4be0f60482291a9452ec7e6f7d68"
EXPECTED_MODEL_SHA256 = "9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb"
EXPECTED_MODEL_FILES = {
    "config.json": "18b4278227eb4ddc2d755fbc0a65b958b3c0ca94c7e67a4418d318679af38737",
    "pytorch_model.bin": EXPECTED_MODEL_SHA256,
    "distilbert_japanese_tokenizer.py": (
        "4af801fa352072c633fcff2c98f61522c6ca31a6ba4489dd2bc8420f2a144fe9"
    ),
    "tokenizer_config.json": "ce4195ae05d31a092b6267741d734384ce977f6d1e1506dafa150b162c6cd135",
    "tokenization_utils.py": "5da30e63ea90d0aa1fb5af3cb948a471cab719dbce3a880dac8859d4c87979b3",
}


def validate_truncated_loading(info: dict[str, Any]) -> tuple[set[int], list[str]]:
    import re

    if info["missing_keys"]:
        raise ValueError(f"pretrained encoder has missing weights: {info['missing_keys']}")
    unexpected = sorted(map(str, info["unexpected_keys"]))
    dropped = set()
    for key in unexpected:
        match = re.search(r"transformer\.layer\.\{?([0-9, ]+)\}?\.", key)
        if match:
            dropped.update(int(value) for value in re.findall(r"\d+", match.group(1)))
        elif not key.startswith("vocab_"):
            raise ValueError(f"unexpected checkpoint key outside dropped layers/MLM head: {key}")
    if dropped != {4, 5}:
        raise ValueError(f"expected exactly pretrained layers 4 and 5 to be omitted; got {dropped}")
    return dropped, unexpected


def source_group_folds(
    rows: list[dict[str, Any]], seed: int = 17
) -> list[tuple[list[int], list[int], str]]:
    sources = sorted({str(row["source"]) for row in rows})
    if len(sources) < 3:
        raise ValueError("at least three source groups are required")
    random.Random(seed).shuffle(sources)
    folds = []
    for source in sources:
        validation = [i for i, row in enumerate(rows) if str(row["source"]) == source]
        train = [i for i, row in enumerate(rows) if str(row["source"]) != source]
        assert_disjoint_groups_and_content(
            [rows[i] for i in train], [rows[i] for i in validation], "source-group fold"
        )
        if not train or not validation:
            raise ValueError("empty source-group fold")
        folds.append((train, validation, source))
    return folds


def configure_upper_layers(model: nn.Module, train_last_layers: int) -> int:
    layers = getattr(getattr(model, "transformer", None), "layer", None)
    if layers is None or not 0 < train_last_layers < len(layers):
        raise ValueError("expected a proper upper-layer subset of DistilBERT")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for layer in layers[-train_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise ValueError("upper-layer adaptation selected no parameters")
    return trainable


def encode_all(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "distilbert" or int(config.n_layers) != 6:
        raise ValueError("the pinned six-layer LINE DistilBERT checkpoint is required")
    config.n_layers = 4
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=False, trust_remote_code=True
    )
    model, info = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        output_loading_info=True,
    )
    dropped, unexpected = validate_truncated_loading(info)
    observed_model_files = {}
    for name, expected_hash in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"pinned LINE DistilBERT file mismatch: {name}")
        observed_model_files[name] = expected_hash
    model.to(device).eval()
    all_first: list[str] = []
    all_second: list[str] = []
    for row in rows:
        context = f"{QUESTION}\n文: {row['sentence']}"
        for option in OPTIONS:
            all_first.append(context)
            all_second.append(option)
    encoded_batches = []
    max_observed = 0
    for start in range(0, len(all_first), batch_size):
        end = start + batch_size
        raw = tokenizer(
            all_first[start:end], all_second[start:end], truncation=False, padding=False
        )
        max_observed = max(max_observed, max(map(len, raw["input_ids"]), default=0))
        if any(len(x) > max_length for x in raw["input_ids"]):
            raise ValueError("candidate pair exceeds the fixed 128-token cap")
        batch = tokenizer(
            all_first[start:end],
            all_second[start:end],
            truncation=False,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if int(batch["input_ids"].max()) >= model.get_input_embeddings().num_embeddings:
            raise ValueError("token ID exceeds the encoder embedding table")
        encoded_batches.append((batch["input_ids"], batch["attention_mask"]))
    input_ids = torch.cat([item[0] for item in encoded_batches]).reshape(len(rows), 2, max_length)
    attention = torch.cat([item[1] for item in encoded_batches]).reshape(len(rows), 2, max_length)
    targets = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    with torch.inference_mode():
        features = []
        for start in range(0, len(input_ids), batch_size // 2):
            stop = start + batch_size // 2
            ids = input_ids[start:stop].reshape(-1, max_length).to(device)
            mask = attention[start:stop].reshape(-1, max_length).to(device)
            features.append(
                model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float().cpu()
            )
        cls = torch.cat(features).reshape(len(rows), 2, -1).contiguous()
    meta = {
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": EXPECTED_MODEL_SHA256,
        "verified_model_file_hashes": observed_model_files,
        "encoder_parameters": sum(p.numel() for p in model.parameters()),
        "layer_count": 4,
        "omitted_pretrained_layers": [4, 5],
        "missing_keys": list(info["missing_keys"]),
        "unexpected_keys": unexpected,
        "omitted_layer_indices": sorted(dropped),
        "max_untruncated_pair_tokens": max_observed,
        "truncated_pairs": 0,
        "input_ids_shape": list(input_ids.shape),
    }
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return input_ids, attention, {"targets": targets, "frozen_cls": cls, **meta}


def _scores_frozen(
    features: Tensor, scorer: nn.Module, device: torch.device, batch_decisions: int
) -> Tensor:
    scorer.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_decisions):
            outputs.append(scorer(features[start : start + batch_decisions].to(device)).cpu())
    return torch.cat(outputs)


def _fit_frozen(
    features: Tensor,
    targets: Tensor,
    train_indices: list[int],
    validation_indices: list[int],
    *,
    seed: int,
    epochs: int,
    batch_decisions: int,
    device: torch.device,
) -> tuple[CandidateScorer, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    scorer = CandidateScorer(int(features.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = list(train_indices)
        rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), batch_decisions):
            idx = torch.tensor(order[start : start + batch_decisions], dtype=torch.long)
            logits = scorer(features[idx].to(device))
            loss = F.cross_entropy(logits, targets[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _scores_frozen(features[validation_indices], scorer, device, batch_decisions)
        val_metrics = metrics(val_logits, targets[validation_indices])
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **val_metrics})
    return scorer, history


def _load_adapted_model(
    model_dir: Path, device: torch.device, seed: int
) -> tuple[nn.Module, nn.Module, int]:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.n_layers = 4
    model, info = AutoModel.from_pretrained(
        model_dir, config=config, local_files_only=True, output_loading_info=True
    )
    validate_truncated_loading(info)
    for name, expected_hash in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"pinned LINE DistilBERT file mismatch: {name}")
    trainable = configure_upper_layers(model, 2)
    # Match the readout initialization used by the frozen arm; model loading
    # itself consumes random draws even though pretrained weights overwrite them.
    torch.manual_seed(seed)
    scorer = CandidateScorer(int(config.hidden_size))
    model.to(device).train()
    scorer.to(device).train()
    return model, scorer, trainable


def _fit_adapted_fold(
    model_dir: Path,
    input_ids: Tensor,
    attention: Tensor,
    targets: Tensor,
    train_indices: list[int],
    validation_indices: list[int],
    *,
    seed: int,
    epochs: int,
    batch_decisions: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    artifact: Path,
) -> tuple[Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, scorer, trainable = _load_adapted_model(model_dir, device, seed)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad] + list(
        scorer.parameters()
    )
    total_encoder_parameters = sum(p.numel() for p in model.parameters())
    scorer_parameters = sum(p.numel() for p in scorer.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(train_indices)
        rng.shuffle(order)
        model.train()
        model.embeddings.eval()
        for frozen_layer in model.transformer.layer[:2]:
            frozen_layer.eval()
        scorer.train()
        losses, norms = [], []
        for start in range(0, len(order), batch_decisions):
            decision_indices = order[start : start + batch_decisions]
            ix = torch.tensor(decision_indices, dtype=torch.long)
            ids = input_ids[ix].reshape(-1, input_ids.shape[-1]).to(device)
            mask = attention[ix].reshape(-1, attention.shape[-1]).to(device)
            target = targets[ix].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float()
            logits = scorer(hidden).reshape(-1, 2)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if not torch.isfinite(norm) or not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss or gradient norm")
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm.detach()))
        model.eval()
        scorer.eval()
        logits_by_batch = []
        with torch.inference_mode():
            for start in range(0, len(validation_indices), batch_decisions):
                ix = torch.tensor(validation_indices[start : start + batch_decisions])
                ids = input_ids[ix].reshape(-1, input_ids.shape[-1]).to(device)
                mask = attention[ix].reshape(-1, attention.shape[-1]).to(device)
                hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float()
                logits_by_batch.append(scorer(hidden).reshape(-1, 2).cpu())
        val_logits = torch.cat(logits_by_batch)
        val_metrics = metrics(val_logits, targets[validation_indices])
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                "gradient_norm_mean": sum(norms) / len(norms),
                "selection_validation_metrics_observation_only": val_metrics,
            }
        )
    # Fixed terminal epoch, no checkpoint selection and no early stopping.
    state = {
        **{
            f"encoder.{key}": value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
            if key.startswith("transformer.layer.2.") or key.startswith("transformer.layer.3.")
        },
        **{
            f"scorer.{key}": value.detach().cpu().contiguous()
            for key, value in scorer.state_dict().items()
        },
    }
    save_file(state, artifact)
    saved = load_file(artifact, device="cpu")
    if set(saved) != set(state) or any(not torch.equal(saved[k], state[k]) for k in state):
        raise ValueError("saved adapted-state reload did not match the final fold weights")
    full_model_state = model.state_dict()
    encoder_state = {
        k.removeprefix("encoder."): v for k, v in saved.items() if k.startswith("encoder.")
    }
    full_model_state.update(encoder_state)
    model.load_state_dict(full_model_state, strict=True)
    scorer_state = {
        k.removeprefix("scorer."): v for k, v in saved.items() if k.startswith("scorer.")
    }
    scorer.load_state_dict(scorer_state, strict=True)
    model.eval()
    scorer.eval()
    prediction_logits = []
    with torch.inference_mode():
        for start in range(0, len(validation_indices), batch_decisions):
            ix = torch.tensor(validation_indices[start : start + batch_decisions])
            ids = input_ids[ix].reshape(-1, input_ids.shape[-1]).to(device)
            mask = attention[ix].reshape(-1, attention.shape[-1]).to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float()
            prediction_logits.append(scorer(hidden).reshape(-1, 2).cpu())
    replay = torch.cat(prediction_logits)
    if not torch.equal(replay, val_logits):
        raise ValueError("best/final CV prediction replay changed")
    report = {
        "trainable_encoder_parameters": trainable,
        "trainable_readout_parameters": scorer_parameters,
        "trainable_total_parameters": trainable + scorer_parameters,
        "total_four_layer_encoder_parameters": total_encoder_parameters,
        "total_candidate_parameters": total_encoder_parameters + scorer_parameters,
        "epochs_completed": epochs,
        "fixed_terminal_epoch": epochs,
        "early_stopping": False,
        "history": history,
        "heldout_source_metrics": metrics(replay, targets[validation_indices]),
        "prediction_logits": replay.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "fold_adapter_sha256": sha256_file(artifact),
        "fold_adapter_bytes": artifact.stat().st_size,
    }
    del model, scorer, optimizer, trainable_parameters
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return replay, report


def _predictions(
    rows: list[dict[str, Any]], indices: list[int], logits: Tensor
) -> list[dict[str, Any]]:
    probs = logits.softmax(-1)
    return [
        {
            "uid": str(rows[row_index]["uid"]),
            "source": str(rows[row_index]["source"]),
            "target": int(rows[row_index]["label"]),
            "prediction": int(logits[j].argmax()),
            "p_unacceptable": float(probs[j, 0]),
            "p_acceptable": float(probs[j, 1]),
        }
        for j, row_index in enumerate(indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fold-epochs", type=int, default=5)
    parser.add_argument("--batch-decisions", type=int, default=16)
    parser.add_argument("--tokenize-batch-pairs", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--smoke-sources", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=0)
    args = parser.parse_args()
    if (
        args.seed != 17
        or args.fold_epochs != 5
        or args.batch_decisions != 16
        or args.max_length != 128
        or args.learning_rate != 2e-5
        or args.weight_decay != 0.01
    ):
        raise ValueError("fixed seed/epoch/batch/optimizer/length protocol; do not override")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    source_revision = subprocess.run(
        ["git", "-C", str(args.source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != SOURCE_REVISION:
        raise ValueError("unexpected JCoLA source commit")
    train_path = args.source_dir / "data" / "jcola-v1.0" / "in_domain_train-v1.0.json"
    rows = read_jsonl(train_path)
    validate_corpus(rows, "JCoLA in-domain train")
    if (
        len(rows) != 6919
        or sha256_file(train_path)
        != "232787aa7bb11f3c42e455aad59d5f54cf1ea8255e9de79f77840de60e0541bf"
    ):
        raise ValueError("frozen JCoLA train payload changed")
    if len({normalize_sentence(str(r["sentence"])) for r in rows}) != len(rows):
        raise ValueError("duplicate normalized sentence in training source")
    if args.smoke_sources:
        if not 3 <= args.smoke_sources <= len({str(row["source"]) for row in rows}):
            raise ValueError("smoke requires at least three complete source groups")
        sources = sorted({str(row["source"]) for row in rows})[: args.smoke_sources]
        selected = []
        for source in sources:
            for label in (0, 1):
                group = [
                    row
                    for row in rows
                    if str(row["source"]) == source and int(row["label"]) == label
                ]
                if not group:
                    raise ValueError("smoke source lacks one class")
                selected.extend(group[:4])
        rows = selected
    folds = source_group_folds(rows, args.seed)
    if args.fold_limit:
        if not args.smoke_sources:
            raise ValueError("partial fold execution is allowed only for CPU smoke")
        folds = folds[: args.fold_limit]
    if not args.smoke_sources and len(folds) != 10:
        raise ValueError("full experiment requires all ten source-held-out folds")
    source_repo = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        not args.smoke_sources
        and subprocess.run(
            ["git", "-C", str(source_repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ):
        raise ValueError("full experiment requires committed clean source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    started_at = datetime.now().astimezone().isoformat()
    config = {
        "source_revision": source_revision,
        "train_sha256": sha256_file(train_path),
        "model_revision": MODEL_REVISION,
        "model_files": EXPECTED_MODEL_FILES,
        "seed": args.seed,
        "source_group_folds": "one source held out per fold; shuffled sources with Random(17)",
        "fold_epochs": args.fold_epochs,
        "early_stopping": False,
        "batch_decisions": args.batch_decisions,
        "candidate_pair_max_length": args.max_length,
        "tokenize_batch_pairs": args.tokenize_batch_pairs,
        "frozen_baseline": "first-four-layer CLS, 128-tanh scorer, AdamW lr=2e-3, wd=1e-4",
        "adapted_candidate": (
            "four-layer DistilBERT; layers 0-1 frozen; layers 2-3 and scorer trainable"
        ),
        "checkpoint_selection": "none; fixed epoch 5 endpoint",
        "label_weighting": "none; unweighted cross entropy",
        "device": str(device),
        "source_commit": source_commit,
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "command_line": shlex.join([sys.executable, *sys.argv]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "started_at": started_at,
        "publisher_in_domain_validation_loaded": False,
        "publisher_ood_development_loaded": False,
        "publisher_test_or_sealed_audit_loaded": False,
    }
    config_bytes = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    (args.output_dir / "effective-config.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    assignment_rows = []
    for fold_index, (_, validation_indices, heldout_source) in enumerate(folds):
        validation_set = set(validation_indices)
        for row_index, row in enumerate(rows):
            assignment_rows.append(
                {
                    "uid": str(row["uid"]),
                    "source": str(row["source"]),
                    "fold": fold_index,
                    "role": "validation" if row_index in validation_set else "train",
                    "heldout_source": heldout_source,
                }
            )
    assignment_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in assignment_rows
    ).encode("utf-8")
    (args.output_dir / "fold-assignments.jsonl").write_bytes(assignment_bytes)
    assignment_sha = hashlib.sha256(assignment_bytes).hexdigest()
    (args.output_dir / "run-progress.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "source_commit": source_commit,
                "configuration_sha256": config_sha,
                "fold_assignment_sha256": assignment_sha,
                "completed_folds": [],
                "started_at": started_at,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    input_ids, attention, encoded_meta = encode_all(
        args.model_dir,
        rows,
        device=device,
        batch_size=args.tokenize_batch_pairs,
        max_length=args.max_length,
    )
    targets = encoded_meta.pop("targets")
    frozen_cls = encoded_meta.pop("frozen_cls")
    baseline_oof: dict[int, Tensor] = {}
    adapted_oof: dict[int, Tensor] = {}
    fold_reports = []
    for fold_index, (train_indices, validation_indices, source) in enumerate(folds):
        base_scorer, base_history = _fit_frozen(
            frozen_cls,
            targets,
            train_indices,
            validation_indices,
            seed=args.seed,
            epochs=args.fold_epochs,
            batch_decisions=args.batch_decisions,
            device=device,
        )
        base_scores = _scores_frozen(
            frozen_cls[validation_indices], base_scorer, device, args.batch_decisions
        )
        base_adapter = args.output_dir / f"fold-{fold_index:02d}-frozen-readout.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in base_scorer.state_dict().items()},
            base_adapter,
        )
        check = load_file(base_adapter, device="cpu")
        if any(
            not torch.equal(check[k], v.detach().cpu()) for k, v in base_scorer.state_dict().items()
        ):
            raise ValueError("frozen readout save/reload mismatch")
        base_scorer.load_state_dict(check, strict=True)
        replayed_base_scores = _scores_frozen(
            frozen_cls[validation_indices],
            base_scorer,
            device,
            args.batch_decisions,
        )
        if not torch.equal(replayed_base_scores, base_scores):
            raise ValueError("frozen readout output changed after safe-tensor reload")
        for j, row_idx in enumerate(validation_indices):
            baseline_oof[row_idx] = base_scores[j]
        del base_scorer
        adapted_scores, adapted_report = _fit_adapted_fold(
            args.model_dir,
            input_ids,
            attention,
            targets,
            train_indices,
            validation_indices,
            seed=args.seed,
            epochs=args.fold_epochs,
            batch_decisions=args.batch_decisions,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            artifact=args.output_dir / f"fold-{fold_index:02d}-upper2-adaptation.safetensors",
        )
        for j, row_idx in enumerate(validation_indices):
            adapted_oof[row_idx] = adapted_scores[j]
        fold_reports.append(
            {
                "fold": fold_index,
                "heldout_source": source,
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                "train_labels": dict(Counter(str(int(rows[i]["label"])) for i in train_indices)),
                "validation_labels": dict(
                    Counter(str(int(rows[i]["label"])) for i in validation_indices)
                ),
                "frozen_readout_metrics": metrics(base_scores, targets[validation_indices]),
                "frozen_readout_fixed_epoch_history": base_history,
                "frozen_readout_sha256": sha256_file(base_adapter),
                "adapted": adapted_report,
            }
        )
        (args.output_dir / f"fold-{fold_index:02d}-summary.json").write_text(
            json.dumps(fold_reports[-1], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"fold-{fold_index:02d}-oof-predictions.json").write_text(
            json.dumps(
                {
                    "heldout_source": source,
                    "frozen_readout": _predictions(rows, validation_indices, base_scores),
                    "upper_two_layers_adapted": _predictions(
                        rows, validation_indices, adapted_scores
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "run-progress.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "source_commit": source_commit,
                    "configuration_sha256": config_sha,
                    "fold_assignment_sha256": assignment_sha,
                    "completed_folds": [row["heldout_source"] for row in fold_reports],
                    "last_completed_fold": fold_index,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"fold": fold_index, "heldout_source": source, "done": True}), flush=True)
    expected_indices = {
        row_index for _, validation_indices, _ in folds for row_index in validation_indices
    }
    if set(baseline_oof) != set(adapted_oof) or set(baseline_oof) != expected_indices:
        raise ValueError("out-of-fold prediction coverage is incomplete")
    ordered = sorted(baseline_oof)
    base_logits = torch.stack([baseline_oof[i] for i in ordered])
    adapted_logits = torch.stack([adapted_oof[i] for i in ordered])
    oof_targets = targets[ordered]
    predictions = {
        "frozen_readout": _predictions(rows, ordered, base_logits),
        "upper_two_layers_adapted": _predictions(rows, ordered, adapted_logits),
    }
    (args.output_dir / "oof-predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    run = {
        "status": "SMOKE_ONLY" if args.smoke_sources else "COMPLETE_SOURCE_GROUPED_CV",
        "source_revision": source_revision,
        "train_sha256": sha256_file(train_path),
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": EXPECTED_MODEL_SHA256,
        "source_groups": len({str(row["source"]) for row in rows}),
        "rows": len(rows),
        "unique_normalized_sentences": len({normalize_sentence(str(r["sentence"])) for r in rows}),
        "label_counts": dict(Counter(str(int(r["label"])) for r in rows)),
        "input_and_model": encoded_meta,
        "configuration": config,
        "configuration_sha256": config_sha,
        "folds": fold_reports,
        "out_of_fold_coverage": len(ordered),
        "frozen_readout_oof_metrics": metrics(base_logits, oof_targets),
        "upper_two_layers_adapted_oof_metrics": metrics(adapted_logits, oof_targets),
        "runtime_seconds": time.time() - started,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "source_commit": source_commit,
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "command_line": shlex.join([sys.executable, *sys.argv]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "fold_assignment_sha256": assignment_sha,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "output_prediction_sha256": sha256_file(args.output_dir / "oof-predictions.json"),
    }
    (args.output_dir / "run-manifest.json").write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run-progress.json").write_text(
        json.dumps(
            {
                "status": run["status"],
                "source_commit": source_commit,
                "configuration_sha256": config_sha,
                "fold_assignment_sha256": assignment_sha,
                "completed_folds": [row["heldout_source"] for row in fold_reports],
                "finished_at": run["finished_at"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": run["status"],
                "coverage": run["out_of_fold_coverage"],
                "frozen": run["frozen_readout_oof_metrics"],
                "adapted": run["upper_two_layers_adapted_oof_metrics"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
