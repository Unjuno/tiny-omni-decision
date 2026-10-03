from __future__ import annotations

import pytest

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DecisionExample, LicenseProvenance, MediaRef
from tiny_omni_decision.training import (
    DecisionTrainingConfig,
    collate_metadata,
    decision_training_config,
    deterministic_sample_order,
    output_record,
)


def example(
    sample_id: str,
    source: str,
    modality: str = "text",
    media: list[MediaRef] | None = None,
) -> DecisionExample:
    return DecisionExample(
        id=sample_id,
        modality=modality,
        state="state",
        question="question",
        options=["left", "right"],
        target="right",
        media=media or [],
        source=source,
        source_revision="a" * 40,
        source_record_id=sample_id,
        split="train",
        provenance=LicenseProvenance(
            license="MIT",
            commercial_use=True,
            derivative_model_training_allowed=True,
            redistribution_allowed=True,
            trust_status="trusted",
        ),
    )


def test_config_defaults_and_sampling_validation() -> None:
    config = DecisionTrainingConfig()
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    with pytest.raises(ValueError, match="at least one modality"):
        DecisionTrainingConfig(modality_weights={"text": 0.0})
    with pytest.raises(ValueError, match="sampling weights"):
        DecisionTrainingConfig(source_weights={"a": -1.0})


def test_repository_training_config_maps_to_explicit_fields() -> None:
    raw = load_structured_file("configs/decision/e2b_qat_lora.yaml")
    config = decision_training_config(raw)
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    assert (config.cross_entropy_weight, config.brier_weight) == (1.0, 0.2)
    assert config.max_steps == 16
    assert set(config.modality_weights) == {"text", "image", "audio", "video"}


def test_corpus_file_sha256_is_reproducible(tmp_path: object) -> None:
    path = tmp_path / "frozen.jsonl"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sampling_is_seeded_balanced_and_reports_consumption() -> None:
    examples = [
        *(example(f"t-{index}", "text-source") for index in range(8)),
        *(
            example(
                f"i-{index}", "image-source", "image", [MediaRef(kind="image", uri=f"x:{index}")]
            )
            for index in range(2)
        ),
    ]
    policy = {"text": 1.0, "image": 1.0}
    first, counts = deterministic_sample_order(examples, seed=9, limit=6, modality_weights=policy)
    second, _ = deterministic_sample_order(examples, seed=9, limit=6, modality_weights=policy)
    assert [item.id for item in first] == [item.id for item in second]
    assert counts == {"image:image-source": 2, "text:text-source": 4}


def test_collation_metadata_and_eval_serialization() -> None:
    item = example("one", "source")
    metadata = collate_metadata(item, ["A", "B"])
    assert metadata["target_index"] == 1
    assert metadata["sample_id"] == item.id
    record = output_record(item, [1.5, 2.5], [0.2, 0.8])
    assert record["prediction"] == 1
    assert record["target"] == 1
    assert record["option_logits"] == [1.5, 2.5]
