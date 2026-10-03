from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_math import label_token_ids_from_prompt, prompt_for_decision
from .schema import DecisionExample


class DecisionTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=17, ge=0)
    max_train_examples: int = Field(default=512, ge=1)
    max_eval_examples: int = Field(default=256, ge=1)
    max_steps: int = Field(default=100, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    checkpoint_interval: int = Field(default=25, ge=1)
    evaluation_interval: int = Field(default=25, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    max_sequence_length: int = Field(default=1024, ge=32)
    ece_bins: int = Field(default=15, ge=1)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    cross_entropy_weight: float = Field(default=1.0, ge=0)
    brier_weight: float = Field(default=0.2, ge=0)
    modality_weights: dict[str, float] = Field(
        default_factory=lambda: {"text": 1.0, "image": 1.0, "audio": 1.0, "video": 1.0}
    )
    source_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sampling_weights(self) -> DecisionTrainingConfig:
        if any(
            value < 0 for value in [*self.modality_weights.values(), *self.source_weights.values()]
        ):
            raise ValueError("sampling weights must be nonnegative")
        if not any(value > 0 for value in self.modality_weights.values()):
            raise ValueError("at least one modality weight must be positive")
        return self


def decision_training_config(raw: dict[str, Any]) -> DecisionTrainingConfig:
    training = raw.get("training", {})
    loss = raw.get("loss", {})
    sampling = raw.get("sampling", {})
    return DecisionTrainingConfig.model_validate(
        {
            "seed": training.get("seed", 17),
            "max_train_examples": training.get("max_train_examples", 512),
            "max_eval_examples": training.get("max_eval_examples", 256),
            "max_steps": training.get("max_steps", 100),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
            "checkpoint_interval": training.get("checkpoint_interval", 25),
            "evaluation_interval": training.get("evaluation_interval", 25),
            "learning_rate": training.get("learning_rate", 1e-4),
            "max_sequence_length": training.get("max_sequence_length", 1024),
            "ece_bins": training.get("ece_bins", 15),
            "lora_rank": training.get("rank", 16),
            "lora_alpha": training.get("alpha", 32),
            "lora_dropout": training.get("dropout", 0.05),
            "cross_entropy_weight": loss.get("cross_entropy", 1.0),
            "brier_weight": loss.get("brier", 0.2),
            "modality_weights": sampling.get(
                "modality_weights", {"text": 1.0, "image": 1.0, "audio": 1.0, "video": 1.0}
            ),
            "source_weights": sampling.get("source_weights", {}),
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
) -> tuple[list[DecisionExample], dict[str, int]]:
    """Weighted deterministic round-robin over modality/source buckets."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    source_weights = source_weights or {}
    buckets: dict[str, list[DecisionExample]] = defaultdict(list)
    for example in examples:
        buckets[sample_key(example)].append(example)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.id)
        rng.shuffle(bucket)
    keys = sorted(buckets)
    weights = {
        key: modality_weights.get(key.split(":", 1)[0], 0.0)
        * source_weights.get(key, source_weights.get(key.split(":", 1)[1], 1.0))
        for key in keys
    }
    if not any(weight > 0 for weight in weights.values()):
        raise ValueError("no examples remain after applying modality/source sampling weights")
    consumed: Counter[str] = Counter()
    selected: list[DecisionExample] = []
    target_count = min(limit, sum(len(bucket) for bucket in buckets.values()))
    while len(selected) < target_count:
        eligible = [key for key in keys if buckets[key] and weights[key] > 0]
        if not eligible:
            break
        # Weighted fair scheduling: each bucket accrues its configured share each round.
        key = max(
            eligible, key=lambda item: (weights[item] * (len(selected) + 1) - consumed[item], item)
        )
        selected.append(buckets[key].pop())
        consumed[key] += 1
    return selected, dict(sorted(consumed.items()))


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
        modality_payload["videos_kwargs"] = {"num_frames": 4}
    encoded = processor(text=rendered, return_tensors="pt", **modality_payload)
    decision_position = int(encoded["input_ids"].shape[-1]) - 1
    target_index = example.options.index(example.target)
    return dict(encoded), option_ids, decision_position, target_index


def resolve_decoder_lora_targets(model: Any) -> list[str]:
    import torch

    candidates = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    )
    if not candidates:
        raise ValueError(
            "loaded pinned model exposes no verified decoder q_proj/v_proj Linear modules"
        )
    return candidates


def output_record(
    example: DecisionExample,
    logits: list[float],
    probabilities: list[float],
) -> dict[str, Any]:
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
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
