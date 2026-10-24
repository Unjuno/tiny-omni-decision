"""Train and evaluate one tiny readout on frozen Perception Test AV features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tiny_omni_decision.perception_source_candidate_readout import (
    build_feature_examples,
    pool_audio_features,
    pool_video_candidate_features,
)

SEED = 17
EPOCHS = 80
BATCH_SIZE = 32
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_DELTA = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def input_artifact_hashes(args):
    return {
        "feature_plan": sha256_file(args.plan),
        "train_labels": sha256_file(args.train_labels),
        "development_labels": sha256_file(args.development_labels),
        "train_video_features": sha256_file(
            args.train_video_features / "candidate_frame_roi_features.npy"
        ),
        "train_video_feature_index": sha256_file(
            args.train_video_features / "feature_index.jsonl"
        ),
        "train_video_feature_result": sha256_file(args.train_video_features / "result.json"),
        "train_audio_features": sha256_file(
            args.train_audio_features / "audio_window_features.npy"
        ),
        "train_audio_window_index": sha256_file(
            args.train_audio_features / "unique_window_index.jsonl"
        ),
        "train_audio_record_references": sha256_file(
            args.train_audio_features / "record_window_references.jsonl"
        ),
        "train_audio_feature_result": sha256_file(args.train_audio_features / "result.json"),
        "development_video_features": sha256_file(
            args.development_video_features / "candidate_frame_roi_features.npy"
        ),
        "development_video_feature_index": sha256_file(
            args.development_video_features / "feature_index.jsonl"
        ),
        "development_video_feature_result": sha256_file(
            args.development_video_features / "result.json"
        ),
        "development_audio_features": sha256_file(
            args.development_audio_features / "audio_window_features.npy"
        ),
        "development_audio_window_index": sha256_file(
            args.development_audio_features / "unique_window_index.jsonl"
        ),
        "development_audio_record_references": sha256_file(
            args.development_audio_features / "record_window_references.jsonl"
        ),
        "development_audio_feature_result": sha256_file(
            args.development_audio_features / "result.json"
        ),
        "runner": sha256_file(Path(__file__).resolve()),
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_examples(plan_path: Path, corpus_split_path: Path, video_root: Path, audio_root: Path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    split_name = "train" if corpus_split_path.name == "train.jsonl" else "development"
    plan_rows = plan[split_name]["records"]
    labels = read_jsonl(corpus_split_path)
    labels_by_id = {row["record_id"]: row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError(f"duplicate split label IDs in {corpus_split_path}")
    if any(row.get("status") != "candidate" or "supervision" not in row for row in labels):
        raise ValueError("fixed split contains a non-candidate or unlabeled row")
    video_index = read_jsonl(video_root / "feature_index.jsonl")
    video_values = np.load(video_root / "candidate_frame_roi_features.npy", mmap_mode="r")
    audio_index = read_jsonl(audio_root / "record_window_references.jsonl")
    audio_values = np.load(audio_root / "audio_window_features.npy", mmap_mode="r")
    visual = pool_video_candidate_features(plan_rows, video_index, video_values)
    source_audio = pool_audio_features(audio_index, audio_values, variant="source")
    offset_audio = pool_audio_features(audio_index, audio_values, variant="offset")
    examples = build_feature_examples(
        plan_rows, labels_by_id, visual, source_audio, offset_audio
    )
    video_ids = {example.video_id for example in examples}
    if len(video_ids) != plan[split_name]["video_group_count"]:
        raise ValueError("feature examples do not cover the frozen video groups")
    return examples, plan


class CrossModalCandidateScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_projection = nn.Linear(1536, HIDDEN_SIZE)
        self.audio_projection = nn.Linear(384, HIDDEN_SIZE)
        self.scorer = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 4, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, video, audio):
        video_hidden = torch.tanh(self.video_projection(video))
        audio_hidden = torch.tanh(self.audio_projection(audio))
        fused = torch.cat(
            (
                video_hidden,
                audio_hidden,
                video_hidden * audio_hidden,
                torch.abs(video_hidden - audio_hidden),
            ),
            dim=-1,
        )
        return self.scorer(fused).squeeze(-1)


def collate_examples(examples, indices, *, audio_variant="source", donor_audio=None):
    batch = [examples[index] for index in indices]
    max_candidates = max(len(row.candidate_ids) for row in batch)
    visual_dim = batch[0].visual_mean_delta.shape[-1]
    audio_dim = batch[0].audio.shape[-1]
    video = np.zeros((len(batch), max_candidates, visual_dim), dtype=np.float32)
    targets = np.zeros((len(batch), max_candidates + 1), dtype=np.bool_)
    candidate_mask = np.zeros((len(batch), max_candidates), dtype=np.bool_)
    audio = np.zeros((len(batch), audio_dim), dtype=np.float32)
    for row_index, example in enumerate(batch):
        candidate_count = len(example.candidate_ids)
        video[row_index, :candidate_count] = example.visual_mean_delta
        candidate_mask[row_index, :candidate_count] = True
        if example.none_visible:
            targets[row_index, max_candidates] = True
        else:
            positions = {candidate_id: i for i, candidate_id in enumerate(example.candidate_ids)}
            for candidate_id in example.target_candidate_ids:
                targets[row_index, positions[candidate_id]] = True
        if audio_variant == "source":
            audio[row_index] = example.audio
        elif audio_variant == "offset":
            audio[row_index] = example.offset_audio
        elif audio_variant == "shuffle":
            if donor_audio is None or example.donor_record_id not in donor_audio:
                raise ValueError(f"shuffle donor missing in fixed split: {example.donor_record_id}")
            audio[row_index] = donor_audio[example.donor_record_id]
        elif audio_variant != "zero":
            raise ValueError(f"unsupported audio variant: {audio_variant}")
    return (
        torch.from_numpy(video),
        torch.from_numpy(audio),
        torch.from_numpy(targets),
        torch.from_numpy(candidate_mask),
        batch,
    )


def score_batch(model, video, audio, candidate_mask, *, video_mode="joint"):
    if video_mode == "zero":
        video = torch.zeros_like(video)
    elif video_mode == "reverse_temporal":
        video = video.clone()
        video[..., 768:] *= -1
    elif video_mode != "joint":
        raise ValueError(f"unsupported video mode: {video_mode}")
    expanded_audio = audio[:, None, :].expand(-1, video.shape[1], -1)
    candidate_logits = model(video, expanded_audio)
    none_video = torch.zeros((video.shape[0], 1, video.shape[-1]), dtype=video.dtype)
    none_logits = model(none_video, audio[:, None, :])
    candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e9)
    return torch.cat((candidate_logits, none_logits), dim=1)


def grouped_nll(logits, targets):
    log_probabilities = torch.log_softmax(logits, dim=1)
    return -torch.logsumexp(log_probabilities.masked_fill(~targets, -torch.inf), dim=1)


def expected_calibration_error(confidences, outcomes, bins=10):
    if not confidences:
        return 0.0
    confidence = np.asarray(confidences, dtype=np.float64)
    outcome = np.asarray(outcomes, dtype=np.float64)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = (confidence >= lower) & (
            confidence < upper if bin_index < bins - 1 else confidence <= upper
        )
        if selected.any():
            error += selected.mean() * abs(confidence[selected].mean() - outcome[selected].mean())
    return float(error)


def average_precision(scores, positives):
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked_positive = positives[order]
    precision = np.cumsum(ranked_positive) / (np.arange(len(order)) + 1)
    return float(precision[ranked_positive].mean())


def evaluate(model, examples, *, audio_mode="source", video_mode="joint", donor_audio=None):
    model.eval()
    per_record = []
    nlls, briers, confidences, correctness, aps = [], [], [], [], []
    visible_correct, visible_total, none_correct, none_total = 0, 0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(examples), BATCH_SIZE):
            indices = list(range(start, min(start + BATCH_SIZE, len(examples))))
            video, audio, targets, mask, batch = collate_examples(
                examples, indices, audio_variant=audio_mode, donor_audio=donor_audio
            )
            logits = score_batch(model, video, audio, mask, video_mode=video_mode)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            nll_values = grouped_nll(logits, targets).cpu().numpy()
            target_array = targets.numpy()
            for local, example in enumerate(batch):
                positive = target_array[local]
                probability = probs[local]
                pred_index = int(np.argmax(probability))
                is_correct = bool(positive[pred_index])
                positive_mass = float(probability[positive].sum())
                target_distribution = positive.astype(np.float64) / int(positive.sum())
                nlls.append(float(nll_values[local]))
                briers.append(float(np.square(probability - target_distribution).sum()))
                confidences.append(float(probability[pred_index]))
                correctness.append(is_correct)
                candidate_scores = probability[: len(example.candidate_ids)]
                candidate_positive = np.asarray(
                    [
                        candidate_id in example.target_candidate_ids
                        for candidate_id in example.candidate_ids
                    ]
                )
                if candidate_positive.any():
                    aps.append(average_precision(candidate_scores, candidate_positive))
                    visible_total += 1
                    visible_correct += int(is_correct)
                else:
                    none_total += 1
                    none_correct += int(pred_index == len(example.candidate_ids))
                predicted = (
                    "none_visible"
                    if pred_index == len(example.candidate_ids)
                    else example.candidate_ids[pred_index]
                )
                per_record.append(
                    {
                        "record_id": example.record_id,
                        "video_id": example.video_id,
                        "candidate_ids": list(example.candidate_ids),
                        "target_candidate_ids": sorted(example.target_candidate_ids),
                        "none_visible": example.none_visible,
                        "predicted": predicted,
                        "top1_correct": is_correct,
                        "top1_confidence": float(probability[pred_index]),
                        "target_probability_mass": positive_mass,
                        "nll": float(nll_values[local]),
                        "candidate_probabilities": {
                            candidate_id: float(probability[index])
                            for index, candidate_id in enumerate(example.candidate_ids)
                        },
                        "none_visible_probability": float(probability[len(example.candidate_ids)]),
                    }
                )
    if len(per_record) != len(examples):
        raise ValueError("evaluation did not produce one prediction per record")
    return {
        "records": len(examples),
        "none_visible_records": none_total,
        "visible_source_records": visible_total,
        "top1_set_accuracy": float(np.mean(correctness)),
        "none_visible_accuracy": none_correct / none_total if none_total else None,
        "visible_source_hit_at_1": visible_correct / visible_total if visible_total else None,
        "mean_set_nll": float(np.mean(nlls)),
        "mean_multiclass_brier": float(np.mean(briers)),
        "top1_ece_10_bins": expected_calibration_error(confidences, correctness),
        "visible_source_candidate_macro_average_precision": float(np.mean(aps)) if aps else None,
        "predictions": per_record,
    }


def train_epoch(model, examples, order, optimizer):
    model.train()
    weighted_loss = 0.0
    consumed = 0
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start : start + BATCH_SIZE]
        video, audio, targets, mask, _ = collate_examples(examples, indices)
        logits = score_batch(model, video, audio, mask)
        loss = grouped_nll(logits, targets).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite grouped training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite readout gradient norm")
        optimizer.step()
        weighted_loss += float(loss.detach()) * len(indices)
        consumed += len(indices)
    return weighted_loss / consumed


def run(args: argparse.Namespace):
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {args.output}")
    if args.epochs != EPOCHS:
        raise ValueError(f"this frozen experiment requires exactly {EPOCHS} epochs")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_initialized():
        raise RuntimeError("readout experiment must be CPU-only")

    train_examples, plan = load_examples(
        args.plan, args.train_labels, args.train_video_features, args.train_audio_features
    )
    development_examples, _ = load_examples(
        args.plan,
        args.development_labels,
        args.development_video_features,
        args.development_audio_features,
    )
    train_ids = {row.record_id for row in train_examples}
    development_ids = {row.record_id for row in development_examples}
    train_videos = {row.video_id for row in train_examples}
    development_videos = {row.video_id for row in development_examples}
    if train_ids & development_ids or train_videos & development_videos:
        raise ValueError("frozen train/development split overlaps by record or video")
    if not train_examples or not development_examples:
        raise ValueError("fixed split is empty")
    train_audio = {row.record_id: row.audio for row in train_examples}
    development_audio = {row.record_id: row.audio for row in development_examples}
    if any(row.donor_record_id not in train_audio for row in train_examples):
        raise ValueError("training audio-shuffle donor crosses or escapes the frozen split")
    if any(row.donor_record_id not in development_audio for row in development_examples):
        raise ValueError("development audio-shuffle donor crosses or escapes the frozen split")

    input_hashes = input_artifact_hashes(args)
    effective_config = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "selector": "lowest development grouped set-NLL; min_delta=1e-4; no early stop",
        "cross_entropy": (
            "negative log probability mass assigned to any valid candidate "
            "or the none-visible choice"
        ),
        "features": (
            "frozen VideoPrism ROI mean plus chronological last-minus-first; "
            "frozen EfficientAT source-audio mean"
        ),
        "controls": [
            "zero_audio",
            "zero_video",
            "shuffled_audio",
            "offset_audio",
            "reverse_temporal_delta",
        ],
        "train_records": len(train_examples),
        "development_records": len(development_examples),
        "train_video_groups": len(train_videos),
        "development_video_groups": len(development_videos),
        "input_artifact_hashes": input_hashes,
    }
    config_hash = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    model = CrossModalCandidateScorer()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    best_nll = math.inf
    best_epoch = None
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(SEED + epoch).permutation(len(train_examples)).tolist()
        train_nll = train_epoch(model, train_examples, order, optimizer)
        validation = evaluate(model, development_examples)
        row = {
            "epoch": epoch,
            "train_grouped_nll": train_nll,
            "development_grouped_nll": validation["mean_set_nll"],
            "development_top1_set_accuracy": validation["top1_set_accuracy"],
            "development_visible_source_map": validation[
                "visible_source_candidate_macro_average_precision"
            ],
            "elapsed_seconds": time.perf_counter() - timer,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        current_nll = validation["mean_set_nll"]
        if current_nll < best_nll - MIN_DELTA:
            best_nll = current_nll
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("validation selector did not retain a checkpoint")
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    evaluation = {}
    for name, audio_mode, video_mode, donor in (
        ("joint", "source", "joint", development_audio),
        ("zero_audio", "zero", "joint", development_audio),
        ("zero_video", "source", "zero", development_audio),
        ("shuffled_audio", "shuffle", "joint", development_audio),
        ("offset_audio", "offset", "joint", development_audio),
        ("reverse_temporal_delta", "source", "reverse_temporal", development_audio),
    ):
        metrics = evaluate(
            model,
            development_examples,
            audio_mode=audio_mode,
            video_mode=video_mode,
            donor_audio=donor,
        )
        predictions = metrics.pop("predictions")
        evaluation[name] = metrics
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"predictions-{name}.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in predictions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    # The readout scores each candidate independently; verify candidate-order invariance.
    first = development_examples[0]
    permutation = np.random.default_rng(SEED).permutation(len(first.candidate_ids))
    if len(permutation) < 2:
        order_invariance_delta = 0.0
    else:
        video = torch.from_numpy(first.visual_mean_delta[None, :, :])
        audio = torch.from_numpy(first.audio[None, :])
        mask = torch.ones((1, len(permutation)), dtype=torch.bool)
        base_scores = score_batch(model, video, audio, mask)[0, : len(permutation)].numpy()
        permuted_scores = score_batch(
            model, video[:, permutation], audio, mask
        )[0, : len(permutation)].numpy()
        order_invariance_delta = float(
            np.max(np.abs(base_scores[permutation] - permuted_scores))
        )
        if order_invariance_delta > 1e-7:
            raise AssertionError("candidate scores changed under candidate reordering")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output / "best-readout-state.pt")
    torch.save(final_state, args.output / "final-readout-state.pt")
    with (args.output / "training-history.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for row in history:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    model_config = {
        **effective_config,
        "config_sha256": config_hash,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_development_nll": best_nll,
        "candidate_order_max_abs_score_delta": order_invariance_delta,
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - timer,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "CPU",
        "cuda_initialized": torch.cuda.is_initialized(),
        "video_group_split_overlap": 0,
        "sealed_audit_read": False,
        "validation_is_blind": False,
        "status": "PASS_FROZEN_FEATURE_READOUT_DIAGNOSTIC",
        "evaluation": evaluation,
    }
    atomic_json(args.output / "result.json", model_config)
    atomic_json(args.output / "input-hashes.json", input_hashes)
    return model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--train-video-features", type=Path, required=True)
    parser.add_argument("--train-audio-features", type=Path, required=True)
    parser.add_argument("--development-video-features", type=Path, required=True)
    parser.add_argument("--development-audio-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
