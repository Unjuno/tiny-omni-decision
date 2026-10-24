"""Frozen-feature learning and evaluation for CLEVR supplied-option decisions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS, answer_kind, make_options
from tiny_omni_decision.clevr_vqa_readout import (
    ClevrAnswerReadout,
    macro_by_kind,
    option_masked_cross_entropy,
    option_probabilities_batch,
    quality_metrics,
)


def prepare_option_tensors(
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Map source answers to deterministic same-kind choices without changing rows."""
    candidate_indices = torch.full((len(rows), 4), -1, dtype=torch.long, device=device)
    option_counts = torch.empty(len(rows), dtype=torch.long, device=device)
    target_positions = torch.empty(len(rows), dtype=torch.long, device=device)
    metadata: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        filename = row.get("image_filename")
        question_index = row.get("question_index")
        answer = row.get("answer")
        if not isinstance(filename, str) or not isinstance(question_index, int):
            raise ValueError("row needs an image_filename and integer question_index")
        if not isinstance(answer, str):
            raise ValueError("row needs an answer string")
        supplied = row.get("options")
        if supplied is None:
            choices = make_options(
                image_filename=filename,
                question_index=question_index,
                answer=answer,
            )
        elif isinstance(supplied, list) and all(isinstance(value, str) for value in supplied):
            choices = tuple(supplied)
            if not 2 <= len(choices) <= 4 or len(set(choices)) != len(choices):
                raise ValueError("supplied options must contain two to four unique answers")
            if any(answer_kind(choice) != answer_kind(answer) for choice in choices):
                raise ValueError("supplied options must share the target answer kind")
            if answer not in choices:
                raise ValueError("supplied options do not contain the target answer")
        else:
            raise ValueError("options must be a list of answer strings")
        if "target_position" in row and row["target_position"] != choices.index(answer):
            raise ValueError("stored target position does not match supplied options")
        candidate_indices[index, : len(choices)] = torch.tensor(
            [ANSWER_LABELS.index(choice) for choice in choices], device=device
        )
        option_counts[index] = len(choices)
        target_positions[index] = choices.index(answer)
        program = row.get("program")
        outer = row.get("outer_program")
        if not isinstance(outer, str):
            outer = (
                program[-1].get("function", "unknown")
                if isinstance(program, list) and program and isinstance(program[-1], dict)
                else "unknown"
            )
        metadata.append(
            {
                "sample_id": f"{filename}:{question_index}",
                "image_filename": filename,
                "question_index": question_index,
                "question": row.get("question", ""),
                "answer": answer,
                "answer_kind": answer_kind(answer),
                "outer_program": outer,
                "options": list(choices),
                "target_position": choices.index(answer),
            }
        )
    return candidate_indices, option_counts, target_positions, metadata


@torch.inference_mode()
def evaluate_readout(
    model: nn.Module,
    features: torch.Tensor,
    rows: Sequence[dict[str, object]],
    *,
    batch_size: int = 512,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return overall, macro, per-kind and per-program metrics plus each prediction."""
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("features must have one [width] row per source question")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidates, counts, targets, metadata = prepare_option_tensors(rows)
    model.eval()
    probability_parts = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        logits = model(features[start:stop])
        probability_parts.append(
            option_probabilities_batch(logits, candidates[start:stop], counts[start:stop])
        )
    probabilities = torch.cat(probability_parts, dim=0)
    overall = quality_metrics(probabilities, targets)
    indices_by_kind: dict[str, list[int]] = defaultdict(list)
    indices_by_program: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        indices_by_kind[str(row["answer_kind"])].append(index)
        indices_by_program[str(row["outer_program"])].append(index)

    def grouped(groups: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        return {
            key: quality_metrics(probabilities[indices], targets[indices])
            for key, indices in sorted(groups.items())
        }

    per_kind = grouped(indices_by_kind)
    metrics: dict[str, object] = {
        "overall": overall,
        "per_kind": per_kind,
        "macro_by_kind": macro_by_kind(per_kind),
        "per_outer_program": grouped(indices_by_program),
    }
    predictions = []
    for index, row in enumerate(metadata):
        prediction = int(probabilities[index].argmax().item())
        row = dict(row)
        row["prediction_position"] = prediction
        row["prediction"] = row["options"][prediction]  # type: ignore[index]
        row["option_probabilities"] = [
            float(value) for value in probabilities[index, : int(counts[index])].tolist()
        ]
        predictions.append(row)
    return metrics, predictions


def fit_readout(
    train_features: torch.Tensor,
    train_rows: Sequence[dict[str, object]],
    validation_features: torch.Tensor,
    validation_rows: Sequence[dict[str, object]],
    *,
    output_dir: Path,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 17,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Fit a small answer head; preserve validation-best and terminal states."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if train_features.shape != (len(train_rows), 768):
        raise ValueError("train feature matrix must be [train questions, 768]")
    if validation_features.shape != (len(validation_rows), 768) or not validation_rows:
        raise ValueError("non-empty validation features must be [validation questions, 768]")
    if not torch.isfinite(train_features).all() or not torch.isfinite(validation_features).all():
        raise FloatingPointError("feature matrix contains non-finite values")

    output_dir.mkdir(parents=True)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = ClevrAnswerReadout().to(target_device)
    train_x = train_features.to(target_device)
    val_x = validation_features.to(target_device)
    train_candidates, train_counts, train_targets, _ = prepare_option_tensors(
        train_rows, device=target_device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, object]] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator)
        loss_total = 0.0
        count_total = 0
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[ids])
            loss = option_masked_cross_entropy(
                logits, train_candidates[ids], train_counts[ids], train_targets[ids]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(ids)
            count_total += len(ids)
        val_metrics, predictions = evaluate_readout(model, val_x, validation_rows)
        macro = val_metrics["macro_by_kind"]
        key = (float(macro["nll"]), float(macro["brier"]))  # type: ignore[index]
        epoch_row = {
            "epoch": epoch,
            "train_option_ce": loss_total / count_total,
            "validation": val_metrics,
            "selected": False,
        }
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            epoch_row["selected"] = True
            torch.save(model.state_dict(), output_dir / "best-readout.pt")
            (output_dir / "best-predictions.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in predictions
                ),
                encoding="utf-8",
            )
        history.append(epoch_row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    torch.save(model.state_dict(), output_dir / "terminal-readout.pt")
    terminal_metrics, terminal_predictions = evaluate_readout(
        model, val_x, validation_rows
    )
    (output_dir / "terminal-predictions.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in terminal_predictions
        ),
        encoding="utf-8",
    )
    return {
        "best_epoch": best_epoch,
        "epochs_completed": epochs,
        "best_validation": history[best_epoch - 1]["validation"],
        "terminal_validation": terminal_metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
