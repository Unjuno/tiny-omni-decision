"""Train only BERT-Mini's final encoder layer on the grouped CSQA train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from audit_commonsenseqa_train import (
    SPLIT_SEED,
    answer_alias_distribution,
    audit_train_parquet,
    normalize,
    stable_split,
)
from fit_commonsenseqa_bert_mini import (
    CHOICE_LABELS,
    EXPECTED_ENCODER_PARAMETERS,
    MODEL_ID,
    MODEL_REVISION,
    _sha256,
    _verify_model_files,
    evaluate_logits,
    soft_target_nll,
)
from safetensors.torch import load_file, save_file

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 3
DEFAULT_MAX_LENGTH = 128
ENCODER_LR = 2e-5
SCORER_LR = 0.01
WEIGHT_DECAY = 0.01


def configure_last_layer_trainable(model: torch.nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    final_layer = model.encoder.layer[-1]
    for parameter in final_layer.parameters():
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _parameter_hash(model: torch.nn.Module, trainable: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tokenize_rows(
    rows: list[dict[str, Any]], model_dir: Path, max_length: int
) -> tuple[dict[str, torch.Tensor], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    questions = [str(row["question"]) for row in rows]
    choices = [str(choice) for row in rows for choice in row["choices"]["text"]]
    repeated_questions = [question for question in questions for _ in range(5)]
    untruncated = tokenizer(
        repeated_questions,
        choices,
        truncation=False,
        add_special_tokens=True,
    )
    truncated_pairs = sum(len(ids) > max_length for ids in untruncated["input_ids"])
    encoded = tokenizer(
        repeated_questions,
        choices,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        key: value.reshape(len(rows), 5, max_length)
        for key, value in encoded.items()
    }, truncated_pairs


def _targets(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor(
        [
            [
                answer_alias_distribution(row).get(label, 0.0)
                for label in CHOICE_LABELS
            ]
            for row in rows
        ],
        dtype=torch.float32,
    )


def _forward_batch(
    model: torch.nn.Module,
    scorer: torch.nn.Linear,
    encoded: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_inputs: dict[str, torch.Tensor] = {}
    for key, tensor in encoded.items():
        selected = tensor[indices]
        batch_inputs[key] = selected.reshape(-1, selected.shape[-1])
    output = model(**batch_inputs)
    return scorer(output.last_hidden_state[:, 0, :]).reshape(len(indices), 5)


def _evaluate(
    model: torch.nn.Module,
    scorer: torch.nn.Linear,
    encoded: dict[str, torch.Tensor],
    targets: torch.Tensor,
    concepts: list[str],
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model.eval()
    scorer.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(concepts), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(concepts)))
            outputs.append(
                _forward_batch(model, scorer, encoded, indices)
            )
    logits = torch.cat(outputs)
    return logits, evaluate_logits(logits, targets, concepts)


def _adapter_state(
    model: torch.nn.Module, scorer: torch.nn.Linear
) -> dict[str, torch.Tensor]:
    state = {
        f"last_layer.{name}": value.detach().cpu().contiguous()
        for name, value in model.encoder.layer[-1].state_dict().items()
    }
    state.update(
        {
            f"scorer.{name}": value.detach().cpu().contiguous()
            for name, value in scorer.state_dict().items()
        }
    )
    return state


def _load_adapter(
    path: Path, model: torch.nn.Module, scorer: torch.nn.Linear
) -> None:
    state = load_file(str(path), device="cpu")
    layer_state = {
        name.removeprefix("last_layer."): value
        for name, value in state.items()
        if name.startswith("last_layer.")
    }
    scorer_state = {
        name.removeprefix("scorer."): value
        for name, value in state.items()
        if name.startswith("scorer.")
    }
    model.encoder.layer[-1].load_state_dict(layer_state, strict=True)
    scorer.load_state_dict(scorer_state, strict=True)


def run_experiment(
    train_path: Path,
    model_dir: Path,
    output_dir: Path,
    seed: int = 17,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    cpu_threads: int = 4,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this experiment is CPU-only")
    if not train_path.name.casefold().startswith("train"):
        raise ValueError("only the pinned training shard may be read")
    if epochs != DEFAULT_EPOCHS:
        raise ValueError("this experiment is fixed at three epochs")

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    torch.set_num_threads(cpu_threads)
    model_files = _verify_model_files(model_dir)
    data_audit = audit_train_parquet(train_path)
    import pyarrow.parquet as parquet
    from transformers import AutoModel

    source_rows = parquet.read_table(train_path).to_pylist()
    train_rows = [
        row for row in source_rows if stable_split(str(row["question_concept"])) == "train"
    ]
    validation_rows = [
        row
        for row in source_rows
        if stable_split(str(row["question_concept"])) == "validation"
    ]
    ordered_rows = train_rows + validation_rows
    train_count = len(train_rows)
    train_concepts = [normalize(str(row["question_concept"])) for row in train_rows]
    validation_concepts = [
        normalize(str(row["question_concept"])) for row in validation_rows
    ]
    encoded, truncated_pairs = _tokenize_rows(ordered_rows, model_dir, max_length)
    train_encoded = {key: value[:train_count] for key, value in encoded.items()}
    validation_encoded = {key: value[train_count:] for key, value in encoded.items()}
    train_targets = _targets(train_rows)
    validation_targets = _targets(validation_rows)

    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=False,
        weights_only=True,
    ).cpu()
    encoder_parameters = sum(parameter.numel() for parameter in model.parameters())
    if encoder_parameters != EXPECTED_ENCODER_PARAMETERS:
        raise ValueError(f"unexpected encoder parameter count: {encoder_parameters}")
    trainable_encoder_parameters = configure_last_layer_trainable(model)
    frozen_before_hash = _parameter_hash(model, trainable=False)
    torch.manual_seed(seed)
    scorer = torch.nn.Linear(256, 1)
    scorer_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    final_layer = model.encoder.layer[-1]
    optimizer = torch.optim.AdamW(
        [
            {"params": final_layer.parameters(), "lr": ENCODER_LR},
            {"params": scorer.parameters(), "lr": SCORER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_state: dict[str, torch.Tensor] | None = None
    update_count = 0

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        model.embeddings.eval()
        for layer in model.encoder.layer[:-1]:
            layer.eval()
        final_layer.train()
        scorer.train()
        order = torch.randperm(train_count, generator=generator)
        losses: list[float] = []
        for start in range(0, train_count, batch_size):
            indices = order[start : start + batch_size]
            logits = _forward_batch(model, scorer, train_encoded, indices)
            loss = soft_target_nll(logits, train_targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*final_layer.parameters(), *scorer.parameters()], 1.0
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            optimizer.step()
            losses.append(loss.item())
            update_count += 1

        train_logits, train_metrics = _evaluate(
            model,
            scorer,
            train_encoded,
            train_targets,
            train_concepts,
            batch_size,
        )
        validation_logits, validation_metrics = _evaluate(
            model,
            scorer,
            validation_encoded,
            validation_targets,
            validation_concepts,
            batch_size,
        )
        selector = validation_metrics["macro_concept_nll"]
        record = {
            "epoch": epoch,
            "optimizer_updates_cumulative": update_count,
            "mean_minibatch_train_nll": sum(losses) / len(losses),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "validation_macro_concept_nll_selector": selector,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        final_state = _adapter_state(model, scorer)
        if selector < best_score:
            best_score = selector
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in final_state.items()}

    if best_state is None or final_state is None:
        raise RuntimeError("no candidate checkpoint was produced")
    frozen_after_hash = _parameter_hash(model, trainable=False)
    if frozen_before_hash != frozen_after_hash:
        raise AssertionError("frozen encoder parameters changed during training")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best-adapter.safetensors"
    final_path = output_dir / "final-adapter.safetensors"
    save_file(best_state, str(best_path))
    save_file(final_state, str(final_path))

    model.encoder.layer[-1].load_state_dict(
        {
            key.removeprefix("last_layer."): value
            for key, value in best_state.items()
            if key.startswith("last_layer.")
        },
        strict=True,
    )
    scorer.load_state_dict(
        {
            key.removeprefix("scorer."): value
            for key, value in best_state.items()
            if key.startswith("scorer.")
        },
        strict=True,
    )
    best_logits, best_metrics = _evaluate(
        model,
        scorer,
        validation_encoded,
        validation_targets,
        validation_concepts,
        batch_size,
    )

    reloaded_model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=False,
        weights_only=True,
    ).cpu().eval()
    reloaded_scorer = torch.nn.Linear(256, 1)
    configure_last_layer_trainable(reloaded_model)
    reloaded_scorer.load_state_dict(
        {
            key.removeprefix("scorer."): value
            for key, value in load_file(str(best_path), device="cpu").items()
            if key.startswith("scorer.")
        },
        strict=True,
    )
    _load_adapter(best_path, reloaded_model, reloaded_scorer)
    reloaded_logits, reloaded_metrics = _evaluate(
        reloaded_model,
        reloaded_scorer,
        validation_encoded,
        validation_targets,
        validation_concepts,
        batch_size,
    )
    reload_error = (best_logits - reloaded_logits).abs().max().item()
    if reload_error > 1e-7 or reloaded_metrics != best_metrics:
        raise AssertionError("best adapter reload changed validation outputs")

    predictions_path = output_dir / "validation-predictions.jsonl"
    probabilities = torch.softmax(reloaded_logits, dim=-1)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as sink:
        for index, row in enumerate(validation_rows):
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "target": validation_targets[index].tolist(),
                        "probabilities": probabilities[index].tolist(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[1]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "UNKNOWN"
    config = {
        "seed": seed,
        "epochs": epochs,
        "batch_size_questions": batch_size,
        "max_length": max_length,
        "last_encoder_layer_learning_rate": ENCODER_LR,
        "scorer_learning_rate": SCORER_LR,
        "weight_decay": WEIGHT_DECAY,
        "option_order": "fixed source order; shared scorer is permutation equivariant",
        "selection": "minimum macro-concept validation NLL across exactly three epochs",
        "target": "uniform mass over labels with text identical to the annotated correct answer",
        "trainable_scope": "BERT encoder final layer and shared scalar scorer only",
        "cpu_threads": cpu_threads,
    }
    metadata = {
        "experiment_id": "commonsenseqa-bert-mini-last-layer-v1",
        "source_commit": source_commit,
        "training_script_sha256": _sha256(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_files": model_files,
        "dataset_revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "dataset_train_sha256": data_audit["input"]["sha256"],
        "split_seed": SPLIT_SEED,
        "split_counts": {"train": train_count, "validation": len(validation_rows)},
        "split_id_order_sha256": data_audit["split_id_order_sha256"],
        "data_quality": {
            "rows_with_duplicate_option_text": data_audit[
                "rows_with_duplicate_option_text"
            ],
            "rows_with_correct_answer_aliases": data_audit[
                "rows_with_correct_answer_aliases"
            ],
            "validation_rows_with_alias_target": sum(
                int((row_target > 0).sum().item() > 1)
                for row_target in validation_targets
            ),
            "pairs_truncated": truncated_pairs,
            "candidate_pairs": len(ordered_rows) * 5,
        },
        "effective_config": config,
        "effective_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "encoder_parameters": encoder_parameters,
            "trainable_encoder_parameters": trainable_encoder_parameters,
            "scorer_parameters": scorer_parameters,
            "combined_parameters": encoder_parameters + scorer_parameters,
            "frozen_parameters_unchanged": frozen_before_hash == frozen_after_hash,
            "frozen_parameters_sha256_before": frozen_before_hash,
            "frozen_parameters_sha256_after": frozen_after_hash,
            "optimizer_updates": update_count,
            "epochs_completed": len(history),
            "selected_epoch": best_epoch,
            "reload_max_abs_logit_error": reload_error,
        },
        "validation_metrics": best_metrics,
        "experiment_wall_seconds": time.perf_counter() - started_wall,
        "experiment_cpu_seconds": time.process_time() - started_cpu,
        "artifact_sha256": {
            best_path.name: _sha256(best_path),
            final_path.name: _sha256(final_path),
            predictions_path.name: _sha256(predictions_path),
            history_path.name: _sha256(history_path),
        },
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not args.train.is_file() or not args.model_dir.is_dir():
        raise FileNotFoundError("training shard or pinned local checkpoint is missing")
    result = run_experiment(
        args.train,
        args.model_dir,
        args.output_dir,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(result["validation_metrics"], indent=2, sort_keys=True))
    print(f"selected epoch: {result['runtime']['selected_epoch']}")
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
