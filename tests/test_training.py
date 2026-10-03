from __future__ import annotations

import pytest

from tiny_omni_decision.corpus import comparison_deltas, partition_heldout_records
from tiny_omni_decision.dataset import deterministic_reservoir_sample, sha256_file
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


def test_durable_teacher_config_bounds_small_modality_reuse() -> None:
    raw = load_structured_file("configs/decision/durable_teacher.yaml")
    config = decision_training_config(raw)
    assert config.max_steps == 256
    assert config.max_sample_repeats == 2


def test_durable_heldout_catalog_keeps_official_open_jev_test_sealed() -> None:
    catalog = load_structured_file("manifests/durable-heldout-corpus.yaml")
    roles = {
        item["manifest"]: item.get("heldout_partition")
        for item in catalog["sources"]
        if item.get("include")
    }
    assert roles["candidates/open-jev-validation.yaml"] == "validation"
    assert roles["candidates/open-jev-test.yaml"] == "evaluation"
    assert all(role is not None for role in roles.values())


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


def test_sampling_balances_modalities_before_splitting_sources() -> None:
    image_examples = (
        example(
            f"i-{index}", "image", "image", [MediaRef(kind="image", uri=f"x:{index}")]
        )
        for index in range(20)
    )
    audio_examples = (
        example(
            f"a-{index}", "audio", "audio", [MediaRef(kind="audio", uri=f"y:{index}")]
        )
        for index in range(20)
    )
    video_examples = (
        example(
            f"v-{index}", "video", "video", [MediaRef(kind="video", uri=f"z:{index}")]
        )
        for index in range(20)
    )
    examples = [
        *(example(f"t1-{index}", "text-one") for index in range(20)),
        *(example(f"t2-{index}", "text-two") for index in range(20)),
        *image_examples,
        *audio_examples,
        *video_examples,
    ]
    selected, counts = deterministic_sample_order(
        examples,
        seed=17,
        limit=40,
        modality_weights={"text": 1, "image": 1, "audio": 1, "video": 1},
    )
    modality_counts = {
        modality: sum(
            count for bucket, count in counts.items() if bucket.startswith(f"{modality}:")
        )
        for modality in ("text", "image", "audio", "video")
    }
    assert modality_counts == {"text": 10, "image": 10, "audio": 10, "video": 10}
    assert counts["text:text-one"] == counts["text:text-two"] == 5
    assert len(selected) == 40


def test_sampling_caps_reuse_to_keep_small_modalities_in_weighted_mix() -> None:
    examples = [
        *(example(f"t-{index}", "text-source") for index in range(20)),
        *(
            example(
                f"v-{index}", "video-source", "video", [MediaRef(kind="video", uri=f"v:{index}")]
            )
            for index in range(2)
        ),
    ]

    selected, counts = deterministic_sample_order(
        examples,
        seed=17,
        limit=8,
        modality_weights={"text": 1, "video": 1},
        max_sample_repeats=4,
    )

    video_ids = [item.id for item in selected if item.modality == "video"]
    assert counts == {"text:text-source": 4, "video:video-source": 4}
    assert len(video_ids) == 4
    assert set(video_ids) == {"v-0", "v-1"}
    assert max(video_ids.count(sample_id) for sample_id in set(video_ids)) <= 4


def test_reservoir_sampling_is_seeded_and_not_a_prefix() -> None:
    rows = [{"id": str(index)} for index in range(100)]
    first = deterministic_reservoir_sample(rows, limit=12, seed=71, source_key="source:train")
    again = deterministic_reservoir_sample(rows, limit=12, seed=71, source_key="source:train")
    other = deterministic_reservoir_sample(rows, limit=12, seed=72, source_key="source:train")
    assert first == again
    assert first != other
    assert len(first) == 12
    assert any(int(row["id"]) > 12 for row in first)


def test_heldout_partition_keeps_flattened_records_together() -> None:
    image_examples = [
        example(
            f"image-{index}-{question}",
            "image-source",
            "image",
            [MediaRef(kind="image", uri=f"image:{index}")],
        ).model_copy(update={"source_record_id": f"image-{index}"})
        for index in range(10)
        for question in range(4)
    ]
    validation, evaluation = partition_heldout_records(image_examples, seed=17)
    val_records = {item.source_record_id for item in validation}
    eval_records = {item.source_record_id for item in evaluation}
    assert len(validation) == len(evaluation) == 20
    assert not val_records & eval_records
    assert val_records | eval_records == {f"image-{index}" for index in range(10)}


def test_heldout_partition_preserves_declared_official_splits() -> None:
    examples = [
        *(
            example(f"validation-{index}", "TypeSafeAI/Open-Jev").model_copy(
                update={"split": "validation"}
            )
            for index in range(4)
        ),
        *(
            example(f"test-{index}", "TypeSafeAI/Open-Jev").model_copy(
                update={"split": "test"}
            )
            for index in range(4)
        ),
    ]

    validation, evaluation = partition_heldout_records(
        examples,
        seed=17,
        heldout_partitions={
            ("TypeSafeAI/Open-Jev", "validation"): "validation",
            ("TypeSafeAI/Open-Jev", "test"): "evaluation",
        },
    )

    assert len(validation) == len(evaluation) == 4
    assert {item.split for item in validation} == {"validation"}
    assert {item.split for item in evaluation} == {"test"}


def test_metric_comparison_deltas() -> None:
    before = {
        "all": {
            "accuracy": 0.4,
            "nll": 2.0,
            "brier": 0.8,
            "ece": 0.3,
            "mean_confidence": 0.7,
        }
    }
    after = {
        "all": {
            "accuracy": 0.5,
            "nll": 1.5,
            "brier": 0.6,
            "ece": 0.2,
            "mean_confidence": 0.6,
        }
    }
    delta = comparison_deltas(before, after)["all"]
    assert delta["accuracy"] == pytest.approx(0.1)
    assert delta["nll"] == pytest.approx(-0.5)
    assert delta["mean_confidence"] == pytest.approx(-0.1)


def test_collation_metadata_and_eval_serialization() -> None:
    item = example("one", "source")
    metadata = collate_metadata(item, ["A", "B"])
    assert metadata["target_index"] == 1
    assert metadata["sample_id"] == item.id
    record = output_record(item, [1.5, 2.5], [0.2, 0.8])
    assert record["prediction"] == 1
    assert record["target"] == 1
    assert record["option_logits"] == [1.5, 2.5]
