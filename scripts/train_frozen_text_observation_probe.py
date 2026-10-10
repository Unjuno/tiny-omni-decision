"""Train a small text Decision scorer over persistent, reusable state features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.cache import ObservationFeatureCache
from tiny_omni_decision.dataset import (
    audit_manifest,
    iter_local_rows,
    normalize_jsonl,
    sha256_file,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.observation_identity import text_observation_feature_key
from tiny_omni_decision.schema import DatasetManifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def _load_examples(path: Path, manifest: DatasetManifest) -> list[Any]:
    return list(normalize_jsonl(iter_local_rows(path), manifest, "typed-decisions-synth"))


def _identity_sets(examples: list[Any]) -> tuple[set[str], set[str]]:
    state_ids = {example.source_record_id.split(":", 1)[0] for example in examples}
    content = {
        hashlib.sha256(
            json.dumps(
                {
                    "state": example.state,
                    "question": example.question,
                    "options": sorted(example.options),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        for example in examples
    }
    return state_ids, content


def _assert_disjoint(train: list[Any], validation: list[Any]) -> None:
    train_groups, train_content = _identity_sets(train)
    validation_groups, validation_content = _identity_sets(validation)
    if train_groups & validation_groups or train_content & validation_content:
        raise ValueError("train/validation state or normalized-content overlap detected")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tokenizer_hash(model_path: Path) -> str:
    names = (
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    )
    digest = hashlib.sha256()
    for name in names:
        path = model_path / name
        if path.is_file():
            payload_hash = sha256_file(path)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload_hash.encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _format_parts(example: Any) -> tuple[str, str, list[str]]:
    return (
        f"State: {example.state}",
        f"Question: {example.question}",
        [f"Candidate answer: {option}" for option in example.options],
    )


def _features_for_batch(
    examples: list[Any],
    embeddings: dict[str, Tensor],
    key_ids: dict[tuple[str, str], str],
) -> tuple[Tensor, list[int], list[int]]:
    rows: list[Tensor] = []
    targets: list[int] = []
    widths: list[int] = []
    for example in examples:
        state_text, question_text, option_texts = _format_parts(example)
        state = embeddings[key_ids[("state", state_text)]]
        question = embeddings[key_ids[("question", question_text)]]
        context = (state + question) * 0.5
        options = torch.stack([embeddings[key_ids[("candidate", text)]] for text in option_texts])
        expanded = context.unsqueeze(0).expand(options.shape[0], -1)
        rows.append(
            torch.cat(
                [
                    expanded,
                    options,
                    (expanded - options).abs(),
                    expanded * options,
                ],
                dim=-1,
            )
        )
        targets.append(example.options.index(example.target))
        widths.append(len(option_texts))
    return torch.cat(rows), targets, widths


def _batch_logits(
    scorer: nn.Module,
    examples: list[Any],
    embeddings: dict[str, Tensor],
    key_ids: dict[tuple[str, str], str],
    device: torch.device,
    batch_questions: int,
) -> tuple[list[Tensor], list[int]]:
    scorer.eval()
    logits: list[Tensor] = []
    targets: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_questions):
            chunk = examples[start : start + batch_questions]
            features, chunk_targets, widths = _features_for_batch(chunk, embeddings, key_ids)
            flat_logits = scorer(features.to(device)).float().cpu()
            offset = 0
            for width in widths:
                logits.append(flat_logits[offset : offset + width])
                offset += width
            if offset != flat_logits.numel():
                raise AssertionError("candidate logits do not cover all examples")
            targets.extend(chunk_targets)
    return logits, targets


def _metrics(logits: list[Tensor], targets: list[int], bins: int = 15) -> dict[str, Any]:
    probabilities = [torch.softmax(item.float(), dim=-1) for item in logits]
    correct = torch.tensor(
        [
            int(int(prob.argmax()) == target)
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    nll = torch.tensor(
        [
            -float(prob[target].clamp_min(1e-12).log())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    brier = torch.tensor(
        [
            float((prob - F.one_hot(torch.tensor(target), prob.numel())).square().sum())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([float(prob.max()) for prob in probabilities])
    ece = 0.0
    counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        count = int(mask.sum())
        counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "ece_bin_counts": counts,
    }


def main() -> None:
    args = _args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")

    reference_dir = args.reference_run_dir.resolve()
    reference = json.loads((reference_dir / "run-report.json").read_text(encoding="utf-8"))
    if reference["status"] != "complete_text_only_synthetic_probe":
        raise ValueError("reference run is not the recorded text-only synthetic probe")
    train_path, validation_path = args.train.resolve(), args.validation.resolve()
    model_path = args.model.resolve()
    if sha256_file(train_path) != reference["train_file_sha256"]:
        raise ValueError("train corpus differs from the immutable text probe corpus")
    if sha256_file(validation_path) != reference["validation_file_sha256"]:
        raise ValueError("validation corpus differs from the immutable text probe corpus")
    if sha256_file(model_path / "model.safetensors") != reference["model_weights_sha256"]:
        raise ValueError("MiniLM weights differ from the pinned baseline")
    if args.revision != reference["model_revision"]:
        raise ValueError("MiniLM revision differs from the pinned baseline")

    train_manifest = DatasetManifest.model_validate(
        __import__("yaml").safe_load(args.train_manifest.read_text(encoding="utf-8"))
    )
    validation_manifest = DatasetManifest.model_validate(
        __import__("yaml").safe_load(args.validation_manifest.read_text(encoding="utf-8"))
    )
    if train_manifest.usage != "training" or validation_manifest.usage != "evaluation":
        raise ValueError("explicit training and evaluation manifests are required")
    if (
        train_manifest.dataset_id != validation_manifest.dataset_id
        or train_manifest.revision != validation_manifest.revision
    ):
        raise ValueError("train and validation manifests must pin the same source revision")
    train_license_audit = audit_manifest(train_manifest)
    validation_license_audit = audit_manifest(validation_manifest)
    if train_license_audit["project_policy"] != "ALLOW":
        raise ValueError(f"training manifest is not ALLOW: {train_license_audit}")
    if validation_license_audit["project_policy"] != "ALLOW":
        raise ValueError(f"validation manifest is not ALLOW: {validation_license_audit}")
    train = _load_examples(train_path, train_manifest)
    validation = _load_examples(validation_path, validation_manifest)
    _assert_disjoint(train, validation)
    if (
        len(train) != reference["train_examples"]
        or len(validation) != reference["validation_examples"]
    ):
        raise ValueError("normalized example counts differ from the fixed text probe")

    saved_predictions = [
        json.loads(line)
        for line in (reference_dir / "validation-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validation_ids = [example.id for example in validation]
    if [row["id"] for row in saved_predictions] != validation_ids:
        raise ValueError("validation ID order differs from the original text probe")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(encoder.config.hidden_size)
    encoder_weights_sha256 = sha256_file(model_path / "model.safetensors")
    tokenizer_sha256 = _tokenizer_hash(model_path)
    preprocessing = {
        "max_length": args.max_length,
        "truncation": True,
        "padding": "longest_in_batch",
        "pooling": "attention_masked_mean_v1",
        "output_dtype": "float32",
        "state_prefix": "State: ",
        "question_prefix": "Question: ",
        "candidate_prefix": "Candidate answer: ",
    }
    preprocessing_sha256 = _sha256_bytes(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    source_id, source_revision = train_manifest.dataset_id, train_manifest.revision

    key_for_text = {}
    text_for_key = {}
    role_counts = Counter()
    input_occurrences = 0
    for example in [*train, *validation]:
        state_text, question_text, option_texts = _format_parts(example)
        for role, text in (
            ("state", state_text),
            ("question", question_text),
            *(("candidate", option) for option in option_texts),
        ):
            key = text_observation_feature_key(
                source_id=source_id,
                source_revision=source_revision,
                observation_text=text,
                encoder_id="sentence-transformers/paraphrase-MiniLM-L3-v2",
                encoder_revision=args.revision,
                encoder_weights_sha256=encoder_weights_sha256,
                tokenizer_sha256=tokenizer_sha256,
                preprocessing_sha256=preprocessing_sha256,
                feature_role=role,
                hidden_size=hidden_size,
            )
            role_counts[role] += 1
            input_occurrences += 1
            identity = (role, text)
            existing = key_for_text.get(identity)
            if existing is not None and existing.cache_id != key.cache_id:
                raise ValueError("same text segment produced inconsistent cache keys")
            key_for_text[identity] = key
            prior_text = text_for_key.setdefault(key.cache_id, text)
            if prior_text != text:
                raise ValueError("content hash collision across text cache entries")

    feature_cache = ObservationFeatureCache(output / "feature-cache")
    unique_texts = list(text_for_key.items())
    direct_embeddings: dict[str, Tensor] = {}
    extraction_started = time.perf_counter()
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(unique_texts), args.feature_batch_size):
            batch = unique_texts[start : start + args.feature_batch_size]
            texts = [text for _, text in batch]
            encoded = tokenizer(
                texts,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu()
            if not torch.isfinite(pooled).all():
                raise ValueError("MiniLM emitted non-finite text features")
            for (cache_id, _), feature in zip(batch, pooled, strict=True):
                direct_embeddings[cache_id] = feature.contiguous()
    extraction_seconds = time.perf_counter() - extraction_started

    output.mkdir(parents=True, exist_ok=True)
    put_started = time.perf_counter()
    for _identity, key in key_for_text.items():
        feature_cache.put(key, direct_embeddings[key.cache_id].numpy().tobytes())
    cache_write_seconds = time.perf_counter() - put_started

    reload_started = time.perf_counter()
    cached_embeddings: dict[str, Tensor] = {}
    payload_bytes = 0
    cache_entry_bytes = 0
    for key in key_for_text.values():
        if key.cache_id in cached_embeddings:
            continue
        item = feature_cache.get(key)
        feature = torch.frombuffer(bytearray(item.payload), dtype=torch.float32).clone()
        if feature.numel() != hidden_size or not torch.isfinite(feature).all():
            raise ValueError("persistent text feature has invalid shape or values")
        cached_embeddings[key.cache_id] = feature
        payload_bytes += len(item.payload)
        cache_entry_bytes += item.entry_bytes
    cache_reload_seconds = time.perf_counter() - reload_started
    if set(cached_embeddings) != set(direct_embeddings):
        raise ValueError("persistent cache reload did not restore every unique feature")

    key_ids = {(role, text): key.cache_id for (role, text), key in key_for_text.items()}
    scorer = FrozenFeatureCandidateScorer(hidden_size * 4, hidden_size=128).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    order_rng = random.Random(args.seed)
    history = []
    best_nll = float("inf")
    best_epoch = -1
    best_state = None
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            examples = [train[index] for index in order[start : start + args.batch_questions]]
            features, target_indices, widths = _features_for_batch(
                examples, cached_embeddings, key_ids
            )
            targets = torch.tensor(target_indices, dtype=torch.long, device=device)
            scores = scorer(features.to(device))
            padded = scores.new_full((len(examples), max(widths)), -1e4)
            offset = 0
            for index, width in enumerate(widths):
                padded[index, :width] = scores[offset : offset + width]
                offset += width
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(padded, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_logits, validation_targets = _batch_logits(
            scorer, validation, cached_embeddings, key_ids, device, args.batch_questions
        )
        validation_metrics = _metrics(validation_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        )
        if validation_metrics["nll"] < best_nll:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("validation did not select a readout checkpoint")
    scorer.load_state_dict(best_state)
    cached_logits, validation_targets = _batch_logits(
        scorer, validation, cached_embeddings, key_ids, device, args.batch_questions
    )
    cached_metrics = _metrics(cached_logits, validation_targets)
    direct_logits, direct_targets = _batch_logits(
        scorer, validation, direct_embeddings, key_ids, device, args.batch_questions
    )
    max_logit_delta = max(
        float((left - right).abs().max())
        for left, right in zip(cached_logits, direct_logits, strict=True)
    )
    direct_predictions = [int(logits.argmax()) for logits in direct_logits]
    cached_predictions = [int(logits.argmax()) for logits in cached_logits]
    if direct_targets != validation_targets or direct_predictions != cached_predictions:
        raise ValueError("persistent-cache reload changed validation target order or predictions")

    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "best-readout.pt")
    predictions = []
    for example, logits in zip(validation, cached_logits, strict=True):
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_index = int(probabilities.argmax())
        predictions.append(
            {
                "id": example.id,
                "source_record_id": example.source_record_id,
                "target": example.target,
                "options": example.options,
                "prediction": example.options[predicted_index],
                "probabilities": probabilities.tolist(),
            }
        )
    prediction_path = output / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    train_logits, train_targets = _batch_logits(
        scorer, train, cached_embeddings, key_ids, device, args.batch_questions
    )
    train_metrics = _metrics(train_logits, train_targets)

    state_unique = {
        split: len(
            {key_for_text[("state", _format_parts(example)[0])].cache_id for example in rows}
        )
        for split, rows in (("train", train), ("validation", validation))
    }
    report = {
        "schema_version": 1,
        "status": "complete_text_observation_cache_decision_probe",
        "task_claim": (
            "synthetic English development task only; not a real-world text quality claim"
        ),
        "source_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "source_worktree_dirty": bool(_git_value(repo_root, "status", "--porcelain")),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "experiment_config": {
            "epochs": args.epochs,
            "batch_questions": args.batch_questions,
            "feature_batch_size": args.feature_batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_size": 128,
            "context_fusion": (
                "arithmetic mean of independent state and question masked-mean embeddings"
            ),
            "candidate_features": "[context, option, absolute_difference, elementwise_product]",
            "checkpoint_selection": "minimum validation NLL",
        },
        "dataset": {
            "id": source_id,
            "revision": source_revision,
            "train_manifest": {
                "path": str(args.train_manifest.resolve()),
                "sha256": sha256_file(args.train_manifest.resolve()),
                "license_policy": train_license_audit,
            },
            "validation_manifest": {
                "path": str(args.validation_manifest.resolve()),
                "sha256": sha256_file(args.validation_manifest.resolve()),
                "license_policy": validation_license_audit,
            },
            "train_file_sha256": sha256_file(train_path),
            "validation_file_sha256": sha256_file(validation_path),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_unique_states": len(_identity_sets(train)[0]),
            "validation_unique_states": len(_identity_sets(validation)[0]),
            "train_validation_state_overlap": 0,
            "train_validation_content_overlap": 0,
            "validation_order_sha256": hashlib.sha256(
                "\n".join(validation_ids).encode()
            ).hexdigest(),
            "validation_ids_match_reference": True,
            "validation_targets_match_reference": [row["target"] for row in saved_predictions]
            == [example.target for example in validation],
        },
        "encoder": {
            "id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "revision": args.revision,
            "weights_sha256": encoder_weights_sha256,
            "tokenizer_files_sha256": tokenizer_sha256,
            "preprocessing": preprocessing,
            "preprocessing_sha256": preprocessing_sha256,
            "parameters_frozen": sum(parameter.numel() for parameter in encoder.parameters()),
            "hidden_size": hidden_size,
            "device": str(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None,
        },
        "cache": {
            "schema": "ObservationFeatureCache-v1",
            "feature_key_roles": dict(
                sorted(Counter(key.feature_name for key in key_for_text.values()).items())
            ),
            "segment_occurrences": dict(sorted(role_counts.items())),
            "unique_cache_entries": len(cached_embeddings),
            "unique_state_features": state_unique,
            "state_encoder_calls_avoided": {
                split: len(rows) - state_unique[split]
                for split, rows in (("train", train), ("validation", validation))
            },
            "unique_text_segment_encodes_avoided": input_occurrences - len(cached_embeddings),
            "state_reuse_ratio": {
                split: 1.0 - state_unique[split] / len(rows)
                for split, rows in (("train", train), ("validation", validation))
            },
            "payload_bytes": payload_bytes,
            "entry_bytes": cache_entry_bytes,
            "write_seconds": cache_write_seconds,
            "reload_seconds": cache_reload_seconds,
            "extraction_seconds": extraction_seconds,
            "direct_vs_reloaded_max_abs_logit_delta": max_logit_delta,
            "direct_vs_reloaded_class_predictions_exact": True,
            "cache_path": str(output / "feature-cache"),
        },
        "readout": {
            "trainable_parameters": trainable_parameters,
            "training_seconds": training_seconds,
            "best_epoch": best_epoch,
            "history": history,
            "train_metrics_at_best": train_metrics,
            "validation_metrics_at_best": cached_metrics,
            "baseline_reference_metrics": reference["best_validation"],
            "note": (
                "The new state/question factorization changes the readout inputs; "
                "this is a separate model probe, not cache-only parity with the old scorer."
            ),
        },
        "artifacts": {
            "best_readout_sha256": sha256_file(output / "best-readout.pt"),
            "validation_predictions_sha256": sha256_file(prediction_path),
            "reference_run_report_sha256": sha256_file(reference_dir / "run-report.json"),
            "reference_feature_cache_sha256": reference["feature_cache_sha256"],
        },
        "sealed_audit_loaded": False,
    }
    report_path = output / "run-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output),
                "validation": cached_metrics,
                "unique_cache_entries": len(cached_embeddings),
                "state_encoder_calls_avoided": report["cache"]["state_encoder_calls_avoided"],
                "run_report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
