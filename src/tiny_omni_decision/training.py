from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .corpus import source_asset_identity
from .decision_math import label_token_ids_from_prompt, prompt_for_decision
from .schema import DecisionExample


class DecisionTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=17, ge=0)
    max_train_examples: int = Field(default=512, ge=1)
    selection_eval_examples: int = Field(default=256, ge=1)
    max_steps: int = Field(default=100, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    checkpoint_interval: int = Field(default=25, ge=1)
    evaluation_interval: int = Field(default=25, ge=1)
    early_stopping_patience: int = Field(default=4, ge=1)
    early_stopping_min_delta: float = Field(default=0.0, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    max_gradient_norm: float = Field(default=1.0, gt=0)
    max_sequence_length: int = Field(default=1024, ge=32)
    video_num_frames: int = Field(default=4, ge=1, le=32)
    ece_bins: int = Field(default=15, ge=1)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_policy: Literal["qv", "attention", "decoder_all_linear"] = "qv"
    lr_scheduler: Literal["constant", "cosine"] = "constant"
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    use_rslora: bool = False
    cross_entropy_weight: float = Field(default=1.0, ge=0)
    brier_weight: float = Field(default=0.2, ge=0)
    modality_weights: dict[str, float] = Field(
        default_factory=lambda: {"text": 1.0, "image": 1.0, "audio": 1.0, "video": 1.0}
    )
    source_weights: dict[str, float] = Field(default_factory=dict)
    video_task_weights: dict[str, float] = Field(default_factory=dict)
    max_sample_repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_sampling_weights(self) -> DecisionTrainingConfig:
        known_video_tasks = {
            "temporal_descriptive",
            "explanatory",
            "predictive",
            "counterfactual",
        }
        if self.video_task_weights and (
            any(value < 0 for value in self.video_task_weights.values())
            or not any(value > 0 for value in self.video_task_weights.values())
            or set(self.video_task_weights) - known_video_tasks
        ):
            raise ValueError(
                "video task weights must be nonnegative, include a positive value, "
                "and name supported video reasoning types"
            )
        if any(
            value < 0 for value in [*self.modality_weights.values(), *self.source_weights.values()]
        ):
            raise ValueError("sampling weights must be nonnegative")
        if not any(value > 0 for value in self.modality_weights.values()):
            raise ValueError("at least one modality weight must be positive")
        if self.lr_scheduler == "constant" and self.warmup_ratio != 0:
            raise ValueError("warmup_ratio must be zero when lr_scheduler is constant")
        return self


def decision_training_config(raw: dict[str, Any]) -> DecisionTrainingConfig:
    training = raw.get("training", {})
    loss = raw.get("loss", {})
    sampling = raw.get("sampling", {})
    return DecisionTrainingConfig.model_validate(
        {
            "seed": training.get("seed", 17),
            "max_train_examples": training.get("max_train_examples", 512),
            "selection_eval_examples": training.get("selection_eval_examples", 256),
            "max_steps": training.get("max_steps", 100),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
            "checkpoint_interval": training.get("checkpoint_interval", 25),
            "evaluation_interval": training.get("evaluation_interval", 25),
            "early_stopping_patience": training.get("early_stopping_patience", 4),
            "early_stopping_min_delta": training.get("early_stopping_min_delta", 0.0),
            "learning_rate": training.get("learning_rate", 1e-4),
            "max_gradient_norm": training.get("max_gradient_norm", 1.0),
            "max_sequence_length": training.get("max_sequence_length", 1024),
            "video_num_frames": training.get("video_num_frames", 4),
            "ece_bins": training.get("ece_bins", 15),
            "lora_rank": training.get("rank", 16),
            "lora_alpha": training.get("alpha", 32),
            "lora_dropout": training.get("dropout", 0.05),
            "lora_target_policy": training.get("lora_target_policy", "qv"),
            "lr_scheduler": training.get("lr_scheduler", "constant"),
            "warmup_ratio": training.get("warmup_ratio", 0.0),
            "use_rslora": training.get("use_rslora", False),
            "cross_entropy_weight": loss.get("cross_entropy", 1.0),
            "brier_weight": loss.get("brier", 0.2),
            "modality_weights": sampling.get(
                "modality_weights", {"text": 1.0, "image": 1.0, "audio": 1.0, "video": 1.0}
            ),
            "source_weights": sampling.get("source_weights", {}),
            "video_task_weights": sampling.get("video_task_weights", {}),
            "max_sample_repeats": sampling.get("max_sample_repeats", 1),
        }
    )


def sample_key(example: DecisionExample) -> str:
    return f"{example.modality}:{example.source}"


def deterministic_sample_order(
    examples: list[DecisionExample],
    *,
    seed: int,
    limit: int,
    modality_weights: dict[str, float],
    source_weights: dict[str, float] | None = None,
    max_sample_repeats: int = 1,
    video_task_weights: dict[str, float] | None = None,
) -> tuple[list[DecisionExample], dict[str, int]]:
    """Weighted deterministic round-robin over modality, source, and video task buckets."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    if max_sample_repeats < 1:
        raise ValueError("max_sample_repeats must be at least one")
    source_weights = source_weights or {}
    if any(value < 0 for value in source_weights.values()):
        raise ValueError("sampling weights must be nonnegative")
    if video_task_weights is not None and (
        any(value < 0 for value in video_task_weights.values())
        or not any(value > 0 for value in video_task_weights.values())
    ):
        raise ValueError("video task weights must be nonnegative with at least one positive value")
    buckets: dict[str, list[DecisionExample]] = defaultdict(list)
    task_buckets: dict[str, dict[str, dict[str, list[DecisionExample]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for example in examples:
        key = sample_key(example)
        buckets[key].append(example)
        if example.modality == "video" and video_task_weights is not None:
            typed_bucket = task_buckets[key]
            if example.task_type is None:
                raise ValueError(f"video task sampling needs task_type for {example.id}")
            if example.task_type not in video_task_weights:
                raise ValueError(
                    f"video task type {example.task_type!r} has no configured sampling weight"
                )
            if video_task_weights[example.task_type] > 0:
                question_id = example.task_group_id or example.id
                typed_bucket[example.task_type].setdefault(question_id, []).append(example)
    rng = random.Random(seed)
    task_group_order: dict[str, dict[str, list[str]]] = defaultdict(dict)
    task_group_rank: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for key, bucket in buckets.items():
        if key in task_buckets:
            for task_type, question_buckets in task_buckets[key].items():
                question_ids = sorted(question_buckets)
                rng.shuffle(question_ids)
                task_group_order[key][task_type] = question_ids
                task_group_rank[key][task_type] = {
                    question_id: rank for rank, question_id in enumerate(question_ids)
                }
                for question_bucket in question_buckets.values():
                    question_bucket.sort(key=lambda item: item.id)
                    rng.shuffle(question_bucket)
                    if max_sample_repeats > 1:
                        question_bucket *= max_sample_repeats
                        rng.shuffle(question_bucket)
        else:
            bucket.sort(key=lambda item: item.id)
            rng.shuffle(bucket)
            if max_sample_repeats > 1:
                bucket *= max_sample_repeats
                rng.shuffle(bucket)

    def remaining(key: str) -> int:
        if key in task_buckets:
            return sum(
                len(question_items)
                for question_buckets in task_buckets[key].values()
                for question_items in question_buckets.values()
            )
        return len(buckets[key])

    keys_by_modality: dict[str, list[str]] = defaultdict(list)
    for key in sorted(buckets):
        modality, source = key.split(":", 1)
        weight = source_weights.get(key, source_weights.get(source, 1.0))
        if weight > 0 and remaining(key) > 0:
            keys_by_modality[modality].append(key)
    active_modalities = {
        modality: weight
        for modality, weight in modality_weights.items()
        if weight > 0 and keys_by_modality.get(modality)
    }
    if not active_modalities:
        raise ValueError("no examples remain after applying modality/source sampling weights")
    eligible_keys = [
        key for modality in active_modalities for key in keys_by_modality[modality]
    ]
    consumed: Counter[str] = Counter()
    modality_consumed: Counter[str] = Counter()
    task_consumed: dict[str, Counter[str]] = defaultdict(Counter)
    task_group_consumed: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    selected: list[DecisionExample] = []
    target_count = min(limit, sum(remaining(key) for key in eligible_keys))
    while len(selected) < target_count:
        eligible_modalities = [
            modality
            for modality in sorted(active_modalities)
            if any(remaining(key) for key in keys_by_modality[modality])
        ]
        if not eligible_modalities:
            break
        # First allocate by modality, then split that modality's share across sources.
        modality_weight_total = sum(active_modalities[item] for item in eligible_modalities)
        modality = max(
            eligible_modalities,
            key=lambda item: (
                (len(selected) + 1) * active_modalities[item] / modality_weight_total
                - modality_consumed[item],
                item,
            ),
        )
        eligible_sources = [
            key for key in keys_by_modality[modality] if remaining(key) > 0
        ]
        source_weight_total = sum(
            source_weights.get(item, source_weights.get(item.split(":", 1)[1], 1.0))
            for item in eligible_sources
        )
        key = max(
            eligible_sources,
            key=lambda item: (
                (modality_consumed[modality] + 1)
                * source_weights.get(item, source_weights.get(item.split(":", 1)[1], 1.0))
                / source_weight_total
                - consumed[item],
                item,
            ),
        )
        if key in task_buckets:
            eligible_tasks = [
                task_type
                for task_type, question_buckets in task_buckets[key].items()
                if any(question_buckets.values()) and video_task_weights[task_type] > 0
            ]
            task_weight_total = sum(video_task_weights[item] for item in eligible_tasks)
            task_type = max(
                eligible_tasks,
                key=lambda item: (
                    (consumed[key] + 1) * video_task_weights[item] / task_weight_total
                    - task_consumed[key][item],
                    item,
                ),
            )
            question_buckets = task_buckets[key][task_type]
            eligible_questions = [
                question_id
                for question_id in task_group_order[key][task_type]
                if question_buckets[question_id]
            ]
            question_id = min(
                eligible_questions,
                key=lambda item: (
                    task_group_consumed[key][task_type][item],
                    task_group_rank[key][task_type][item],
                ),
            )
            selected.append(question_buckets[question_id].pop())
            task_consumed[key][task_type] += 1
            task_group_consumed[key][task_type][question_id] += 1
        else:
            selected.append(buckets[key].pop())
        consumed[key] += 1
        modality_consumed[modality] += 1
    return selected, dict(sorted(consumed.items()))


def deterministic_validation_subset(
    examples: list[DecisionExample],
    *,
    seed: int,
    limit: int,
    modalities: tuple[str, ...] = ("text", "image", "audio", "video"),
    video_task_weights: dict[str, float] | None = None,
) -> list[DecisionExample]:
    """Select the same number of independent validation examples per modality."""
    if limit < len(modalities):
        raise ValueError("validation limit must allow at least one example per modality")
    available = {
        modality: [example for example in examples if example.modality == modality]
        for modality in modalities
    }
    missing = [modality for modality, items in available.items() if not items]
    if missing:
        raise ValueError(f"validation corpus is missing modalities: {missing}")
    per_modality = min(limit // len(modalities), *(len(items) for items in available.values()))
    selected: list[DecisionExample] = []
    for modality in modalities:
        items, _ = deterministic_sample_order(
            available[modality],
            seed=seed,
            limit=per_modality,
            modality_weights={modality: 1.0},
            max_sample_repeats=1,
            video_task_weights=video_task_weights,
        )
        selected.extend(items)
    return selected


def sampling_accounting(examples: list[DecisionExample]) -> dict[str, Any]:
    """Measure actual consumed examples, repeats, and unique source assets."""
    consumed_by_source: Counter[str] = Counter()
    consumed_by_modality: Counter[str] = Counter()
    unique_by_source: dict[str, set[tuple[str, str]]] = defaultdict(set)
    unique_by_modality: dict[str, set[tuple[str, str]]] = defaultdict(set)
    assets: set[tuple[str, str, str]] = set()
    assets_by_modality: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    assets_by_source: dict[str, set[str]] = defaultdict(set)
    assets_by_source_modality: dict[str, set[str]] = defaultdict(set)
    sample_occurrences: Counter[tuple[str, str]] = Counter()
    consumed_by_task_type: Counter[str] = Counter()
    unique_by_task_type: dict[str, set[str]] = defaultdict(set)
    unique_questions_by_task_type: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for example in examples:
        source_key = (example.source, example.id)
        asset_key = (example.source, example.modality, source_asset_identity(example))
        sample_occurrences[source_key] += 1
        if example.task_type is not None:
            consumed_by_task_type[example.task_type] += 1
            unique_by_task_type[example.task_type].add(example.id)
            question_group = example.task_group_id or example.id
            unique_questions_by_task_type[example.task_type].add(
                (example.source, question_group)
            )
        consumed_by_source[example.source] += 1
        consumed_by_modality[example.modality] += 1
        unique_by_source[example.source].add(source_key)
        unique_by_modality[example.modality].add(source_key)
        assets.add(asset_key)
        assets_by_modality[example.modality].add(asset_key)
        assets_by_source[example.source].add(asset_key[2])
        assets_by_source_modality[f"{example.source}:{example.modality}"].add(asset_key[2])

    return {
        "samples_consumed": len(examples),
        "unique_examples": len(sample_occurrences),
        "repeated_example_count": sum(count - 1 for count in sample_occurrences.values()),
        "unique_underlying_assets": len(assets),
        "consumed_by_task_type": dict(sorted(consumed_by_task_type.items())),
        "unique_examples_by_task_type": {
            task_type: len(items) for task_type, items in sorted(unique_by_task_type.items())
        },
        "unique_questions_by_task_type": {
            task_type: len(items)
            for task_type, items in sorted(unique_questions_by_task_type.items())
        },
        "consumed_by_source": dict(sorted(consumed_by_source.items())),
        "consumed_by_modality": dict(sorted(consumed_by_modality.items())),
        "unique_examples_by_source": {
            source: len(items) for source, items in sorted(unique_by_source.items())
        },
        "unique_examples_by_modality": {
            modality: len(items) for modality, items in sorted(unique_by_modality.items())
        },
        "unique_underlying_assets_by_modality": {
            modality: len(items) for modality, items in sorted(assets_by_modality.items())
        },
        "unique_underlying_assets_by_source": {
            source: len(items) for source, items in sorted(assets_by_source.items())
        },
        "unique_underlying_assets_by_source_modality": {
            key: len(items) for key, items in sorted(assets_by_source_modality.items())
        },
    }


def aggregate_training_window(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step train statistics over the interval before validation."""
    if not step_records:
        raise ValueError("training window must contain at least one optimizer step")
    total_examples = sum(int(record["train_examples"]) for record in step_records)
    if total_examples < 1:
        raise ValueError("training window must contain at least one consumed example")

    correct_by_modality: Counter[str] = Counter()
    examples_by_modality: Counter[str] = Counter()
    ce_sum_by_modality: defaultdict[str, float] = defaultdict(float)
    for record in step_records:
        for modality, count in record["train_examples_by_modality"].items():
            examples_by_modality[modality] += int(count)
            ce_sum_by_modality[modality] += (
                float(record["train_ce_by_modality"][modality]) * int(count)
            )
        for modality, count in record["train_correct_by_modality"].items():
            correct_by_modality[modality] += int(count)

    total_correct = sum(correct_by_modality.values())
    return {
        "train_ce": sum(
            float(record["cross_entropy"]) * int(record["train_examples"])
            for record in step_records
        )
        / total_examples,
        "train_accuracy": total_correct / total_examples,
        "train_accuracy_by_modality": {
            modality: correct_by_modality[modality] / count
            for modality, count in sorted(examples_by_modality.items())
        },
        "train_ce_by_modality": {
            modality: ce_sum_by_modality[modality] / count
            for modality, count in sorted(examples_by_modality.items())
        },
        "training_examples_in_window": total_examples,
        "microbatches_in_window": sum(
            int(record["microbatches"]) for record in step_records
        ),
        "optimizer_steps_in_window": len(step_records),
    }


def project_runtime_seconds(
    *,
    seconds_per_optimizer_step: float,
    optimizer_steps: int,
    evaluation_interval: int,
    mean_scheduled_evaluation_seconds: float,
    setup_evaluation_seconds: float,
    final_evaluation_seconds: float,
    other_overhead_seconds: float = 0.0,
) -> float:
    """Estimate end-to-end runtime including scheduled and final validation."""
    if optimizer_steps < 1 or evaluation_interval < 1:
        raise ValueError("runtime projection requires positive step and evaluation intervals")
    if any(
        value < 0
        for value in (
            seconds_per_optimizer_step,
            mean_scheduled_evaluation_seconds,
            setup_evaluation_seconds,
            final_evaluation_seconds,
            other_overhead_seconds,
        )
    ):
        raise ValueError("runtime projection durations must be nonnegative")
    scheduled_evaluations = math.ceil(optimizer_steps / evaluation_interval)
    return (
        seconds_per_optimizer_step * optimizer_steps
        + mean_scheduled_evaluation_seconds * scheduled_evaluations
        + setup_evaluation_seconds
        + final_evaluation_seconds
        + other_overhead_seconds
    )


def learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    scheduler: Literal["constant", "cosine"],
    warmup_ratio: float,
) -> float:
    """Return the configured LR multiplier at a zero-based optimizer step."""
    if total_steps < 1 or step < 0:
        raise ValueError(
            "learning-rate schedule requires nonnegative step and positive total_steps"
        )
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if scheduler == "constant":
        if warmup_ratio != 0:
            raise ValueError("warmup_ratio must be zero when scheduler is constant")
        return 1.0
    if scheduler != "cosine":
        raise ValueError(f"unsupported learning-rate scheduler: {scheduler}")

    warmup_steps = math.ceil(total_steps * warmup_ratio)
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def collate_metadata(example: DecisionExample, labels: list[str]) -> dict[str, Any]:
    if len(labels) != len(example.options):
        raise ValueError("one decision label is required per option")
    if example.target not in example.options:
        raise ValueError("target is not in the options")
    return {
        "sample_id": example.id,
        "source": example.source,
        "modality": example.modality,
        "option_labels": labels,
        "target_index": example.options.index(example.target),
        "media_kinds": [reference.kind for reference in example.media],
    }


def processor_inputs_for_example(
    processor: Any,
    example: DecisionExample,
    *,
    data_root: Path,
    video_num_frames: int = 4,
) -> tuple[dict[str, Any], list[int], int, int]:
    """Build actual Gemma4Processor inputs from a metadata record and local media."""
    labels = [chr(ord("A") + index) for index in range(len(example.options))]
    prompt = prompt_for_decision(example, labels)
    modality_payload: dict[str, Any] = {}
    media_line = ""
    if example.modality != "text":
        refs = [reference for reference in example.media if reference.kind == example.modality]
        if len(refs) != 1:
            raise ValueError(
                f"{example.id}: expected exactly one {example.modality} media reference"
            )
        reference = refs[0]
        if not reference.path:
            raise ValueError(f"{example.id}: media is not locally materialized ({reference.uri})")
        media_path = (data_root / reference.path).resolve()
        if data_root.resolve() not in media_path.parents:
            raise ValueError(f"{example.id}: media path escapes the configured data root")
        if not media_path.is_file():
            raise FileNotFoundError(f"{example.id}: local media is missing: {media_path}")
        if example.modality == "image":
            from PIL import Image

            modality_payload["images"] = [Image.open(media_path).convert("RGB")]
            media_line = processor.image_token
        elif example.modality == "audio":
            import wave

            import numpy as np

            with wave.open(str(media_path), "rb") as audio_file:
                channels = audio_file.getnchannels()
                sample_rate = audio_file.getframerate()
                if sample_rate != 16_000:
                    raise ValueError(
                        f"Gemma 4 audio path expects 16 kHz input, got {sample_rate} Hz"
                    )
                frames = audio_file.readframes(audio_file.getnframes())
                waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    waveform = waveform.reshape(-1, channels).mean(axis=1)
            modality_payload["audio"] = [waveform]
            media_line = processor.audio_token
        else:
            modality_payload["videos"] = [str(media_path)]
            media_line = processor.video_token
    if media_line:
        prompt = f"{media_line}\n{prompt}"
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    option_ids, prompt_length = label_token_ids_from_prompt(processor.tokenizer, rendered, labels)
    if example.modality == "video":
        if video_num_frames < 1:
            raise ValueError("video_num_frames must be at least one")
        modality_payload["videos_kwargs"] = {"num_frames": video_num_frames}
    encoded = processor(text=rendered, return_tensors="pt", **modality_payload)
    decision_position = int(encoded["input_ids"].shape[-1]) - 1
    target_index = example.options.index(example.target)
    return dict(encoded), option_ids, decision_position, target_index


def resolve_decoder_lora_targets(
    model: Any,
    *,
    policy: Literal["qv", "attention", "decoder_all_linear"] = "qv",
) -> list[str]:
    import torch

    suffixes_by_policy = {
        "qv": {"q_proj", "v_proj"},
        "attention": {"q_proj", "k_proj", "v_proj", "o_proj"},
        "decoder_all_linear": {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        },
    }
    try:
        selected_suffixes = suffixes_by_policy[policy]
    except KeyError as exc:
        raise ValueError(f"unsupported LoRA target policy: {policy}") from exc

    candidates = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in selected_suffixes
    )
    if not candidates:
        raise ValueError(
            f"loaded pinned model exposes no verified decoder {policy} Linear modules"
        )
    return candidates


def output_record(
    example: DecisionExample,
    logits: list[float],
    probabilities: list[float],
) -> dict[str, Any]:
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    record = {
        "sample_id": example.id,
        "source": example.source,
        "modality": example.modality,
        "target": example.options.index(example.target),
        "option_labels": [chr(ord("A") + index) for index in range(len(example.options))],
        "option_logits": logits,
        "option_probabilities": probabilities,
        "prediction": prediction,
        "confidence": probabilities[prediction],
    }
    if example.task_type is not None:
        record["task_type"] = example.task_type
    if example.task_group_id is not None:
        record["task_group_id"] = example.task_group_id
    return record
