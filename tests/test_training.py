from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tiny_omni_decision import trainer
from tiny_omni_decision.corpus import comparison_deltas, partition_heldout_records
from tiny_omni_decision.dataset import (
    check_train_eval_splits,
    corpus_statistics,
    deterministic_reservoir_sample,
    sha256_file,
)
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DecisionExample, LicenseProvenance, MediaRef
from tiny_omni_decision.training import (
    DecisionTrainingConfig,
    aggregate_training_window,
    collate_metadata,
    decision_training_config,
    deterministic_sample_order,
    learning_rate_multiplier,
    output_record,
    processor_inputs_for_example,
    project_runtime_seconds,
    resolve_decoder_lora_targets,
    sampling_accounting,
)


def available_helper(module_name: str, helper_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"missing required behavior module {module_name}")
    helper = getattr(module, helper_name, None)
    assert callable(helper), f"missing required behavior helper {module_name}.{helper_name}"
    return helper


def example(
    sample_id: str,
    source: str,
    modality: str = "text",
    media: list[MediaRef] | None = None,
    task_type: str | None = None,
    task_group_id: str | None = None,
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
        task_type=task_type,
        task_group_id=task_group_id,
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
    assert (
        config.video_num_frames,
        config.lora_target_policy,
        config.lr_scheduler,
        config.warmup_ratio,
        config.use_rslora,
    ) == (4, "qv", "constant", 0.0, False)
    with pytest.raises(ValueError, match="at least one modality"):
        DecisionTrainingConfig(modality_weights={"text": 0.0})
    with pytest.raises(ValueError, match="sampling weights"):
        DecisionTrainingConfig(source_weights={"a": -1.0})


def test_artifact_corpus_can_use_explicit_repository_media_root(tmp_path: Path) -> None:
    train_path = tmp_path / "artifacts" / "teacher-v2" / "candidate-e" / "train.jsonl"
    media_root = tmp_path / "data"

    assert trainer.resolve_media_root(train_path, media_root) == media_root.resolve()


def test_default_media_root_preserves_processed_corpus_layout(tmp_path: Path) -> None:
    train_path = tmp_path / "data" / "processed" / "teacher-v1" / "train.jsonl"

    assert trainer.resolve_media_root(train_path, None) == (tmp_path / "data").resolve()


def test_teacher_v1_explicit_reference_and_candidate_a_change_only_frames() -> None:
    baseline_raw = load_structured_file("configs/decision/teacher_v1_explicit_reference.yaml")
    candidate_raw = load_structured_file("configs/decision/teacher_v2_candidate_a.yaml")
    baseline = decision_training_config(baseline_raw)
    candidate = decision_training_config(candidate_raw)

    assert baseline_raw["teacher_id"] == "tiny-omni-decision-teacher-v1"
    assert candidate_raw["teacher_id"] == "tiny-omni-decision-teacher-v2-candidate-a"
    assert candidate_raw["reference_teacher_id"] == "tiny-omni-decision-teacher-v1"
    assert baseline.video_num_frames == 4
    assert candidate.video_num_frames == 8
    assert candidate.model_copy(update={"video_num_frames": 4}) == baseline
    assert (candidate.lora_target_policy, candidate.lora_rank) == ("qv", 16)
    assert (candidate.learning_rate, candidate.lr_scheduler, candidate.warmup_ratio) == (
        5e-5,
        "constant",
        0.0,
    )
    assert candidate.use_rslora is False
    assert baseline_raw["sampling"] == candidate_raw["sampling"]


def test_candidate_b_changes_only_decoder_lora_target_policy() -> None:
    candidate_a = decision_training_config(
        load_structured_file("configs/decision/teacher_v2_candidate_a.yaml")
    )
    candidate_b = decision_training_config(
        load_structured_file("configs/decision/teacher_v2_candidate_b.yaml")
    )

    assert candidate_b.lora_target_policy == "decoder_all_linear"
    assert candidate_b.model_copy(update={"lora_target_policy": "qv"}) == candidate_a
    assert candidate_b.video_num_frames == 8
    assert candidate_b.lora_rank == 16
    assert candidate_b.learning_rate == candidate_a.learning_rate == 5e-5
    assert candidate_b.lr_scheduler == candidate_a.lr_scheduler == "constant"
    assert candidate_b.warmup_ratio == candidate_a.warmup_ratio == 0.0
    assert candidate_b.use_rslora is candidate_a.use_rslora is False


def test_candidate_c_changes_only_rank_and_rslora_from_candidate_b() -> None:
    candidate_b = decision_training_config(
        load_structured_file("configs/decision/teacher_v2_candidate_b.yaml")
    )
    candidate_c = decision_training_config(
        load_structured_file("configs/decision/teacher_v2_candidate_c.yaml")
    )

    assert candidate_c.model_copy(update={"lora_rank": 16, "use_rslora": False}) == candidate_b
    assert candidate_c.lora_target_policy == candidate_b.lora_target_policy == "decoder_all_linear"
    assert candidate_c.video_num_frames == candidate_b.video_num_frames == 8
    assert candidate_c.lora_rank == 32
    assert candidate_c.use_rslora is True
    assert candidate_c.learning_rate == candidate_b.learning_rate == 5e-5
    assert candidate_c.lr_scheduler == candidate_b.lr_scheduler == "constant"
    assert candidate_c.warmup_ratio == candidate_b.warmup_ratio == 0.0


def test_candidate_b_cosine_changes_only_the_optimizer_schedule() -> None:
    candidate_b_raw = load_structured_file("configs/decision/teacher_v2_candidate_b.yaml")
    cosine_raw = load_structured_file("configs/decision/teacher_v2_candidate_b_cosine.yaml")
    candidate_b = decision_training_config(candidate_b_raw)
    cosine = decision_training_config(cosine_raw)

    assert cosine == candidate_b.model_copy(
        update={"lr_scheduler": "cosine", "warmup_ratio": 0.03}
    )
    assert cosine.lr_scheduler == "cosine"
    assert cosine.warmup_ratio == 0.03
    assert cosine.learning_rate == candidate_b.learning_rate == 5e-5
    assert cosine.video_num_frames == candidate_b.video_num_frames == 8
    assert cosine.lora_target_policy == candidate_b.lora_target_policy == "decoder_all_linear"
    assert cosine.lora_rank == candidate_b.lora_rank == 16
    assert cosine.use_rslora is candidate_b.use_rslora is False
    for section in ("base_model", "task", "loss", "sampling"):
        assert cosine_raw[section] == candidate_b_raw[section]
    assert cosine_raw["reference_teacher_id"] == candidate_b_raw["reference_teacher_id"]
    b_training = candidate_b_raw["training"]
    cosine_training = cosine_raw["training"]
    assert {
        key: value
        for key, value in cosine_training.items()
        if key not in {"lr_scheduler", "warmup_ratio"}
    } == {
        key: value
        for key, value in b_training.items()
        if key not in {"lr_scheduler", "warmup_ratio"}
    }


def test_windows_base_model_loader_uses_pread_and_restores_transformers_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_omni_decision.trainer as trainer

    modeling_utils = ModuleType("transformers.modeling_utils")
    calls = []

    def fake_safe_open(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        return "opened"

    modeling_utils.safe_open = fake_safe_open
    transformers = ModuleType("transformers")
    transformers.__path__ = []
    transformers.modeling_utils = modeling_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setattr(trainer.sys, "platform", "win32")
    expected_model = object()

    def load_model(repo_id, *, revision, **kwargs):
        assert repo_id == "model/repo"
        assert revision == "revision"
        assert kwargs == {"dtype": "auto"}
        assert modeling_utils.safe_open("weights.safetensors", framework="pt") == "opened"
        return expected_model

    result = trainer._load_pretrained_base(
        load_model, "model/repo", revision="revision", dtype="auto"
    )

    assert result is expected_model
    assert calls == [
        (("weights.safetensors",), {"framework": "pt", "backend": "pread"})
    ]
    assert modeling_utils.safe_open is fake_safe_open


def test_base_model_loader_leaves_non_windows_defaults_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_omni_decision.trainer as trainer

    monkeypatch.setattr(trainer.sys, "platform", "linux")
    calls = []

    def load_model(repo_id, *, revision, **kwargs):
        calls.append((repo_id, revision, kwargs))
        return "loaded"

    result = trainer._load_pretrained_base(
        load_model, "model/repo", revision="revision", dtype="auto"
    )

    assert result == "loaded"
    assert calls == [("model/repo", "revision", {"dtype": "auto"})]


def test_lora_target_policies_are_limited_to_decoder_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinear:
        pass

    target_names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.mm_projector.q_proj",
        "model.vision_tower.layers.0.self_attn.q_proj",
        "model.audio_tower.layers.0.mlp.gate_proj",
    ]
    fake_torch = ModuleType("torch")
    fake_torch.nn = SimpleNamespace(Linear=FakeLinear)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def named_modules(self) -> list[tuple[str, FakeLinear]]:
            return [(name, FakeLinear()) for name in target_names]

    model = FakeModel()
    resolved_by_policy = {
        policy: resolve_decoder_lora_targets(model, policy=policy)
        for policy in ("qv", "attention", "decoder_all_linear")
    }
    assert {name.rsplit(".", 1)[-1] for name in resolved_by_policy["qv"]} == {
        "q_proj",
        "v_proj",
    }
    assert {name.rsplit(".", 1)[-1] for name in resolved_by_policy["attention"]} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert {name.rsplit(".", 1)[-1] for name in resolved_by_policy["decoder_all_linear"]} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert all(
        name.startswith("model.language_model.layers.")
        for names in resolved_by_policy.values()
        for name in names
    )


def test_learning_rate_multiplier_keeps_v1_constant_and_supports_separate_cosine() -> None:
    assert learning_rate_multiplier(0, total_steps=100, scheduler="constant", warmup_ratio=0) == 1
    assert learning_rate_multiplier(100, total_steps=100, scheduler="constant", warmup_ratio=0) == 1
    assert learning_rate_multiplier(
        0, total_steps=100, scheduler="cosine", warmup_ratio=0.03
    ) == pytest.approx(1 / 3)
    assert learning_rate_multiplier(3, total_steps=100, scheduler="cosine", warmup_ratio=0.03) == 1
    assert (
        learning_rate_multiplier(100, total_steps=100, scheduler="cosine", warmup_ratio=0.03)
        == 0
    )
    with pytest.raises(ValueError, match="warmup_ratio must be zero"):
        DecisionTrainingConfig(lr_scheduler="constant", warmup_ratio=0.03)


def test_processor_receives_configured_video_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tiny_omni_decision.training as training_module

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"test video placeholder")
    video_example = example(
        "video-frame-smoke",
        "MIT-IBM/CLEVRER",
        "video",
        [MediaRef(kind="video", path="clip.mp4")],
    )
    monkeypatch.setattr(training_module, "prompt_for_decision", lambda *_: "decision prompt")
    monkeypatch.setattr(
        training_module,
        "label_token_ids_from_prompt",
        lambda *_: ([11, 12], 5),
    )

    class RecordingProcessor:
        video_token = "<video>"
        tokenizer = object()

        def __init__(self) -> None:
            self.payload: dict[str, object] = {}

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "rendered prompt"

        def __call__(self, **kwargs: object) -> dict[str, object]:
            self.payload = kwargs
            return {"input_ids": SimpleNamespace(shape=(1, 19))}

    processor = RecordingProcessor()
    inputs, _, _, _ = processor_inputs_for_example(
        processor, video_example, data_root=tmp_path, video_num_frames=8
    )

    assert processor.payload["videos_kwargs"] == {"num_frames": 8}
    assert inputs["input_ids"].shape[-1] == 19


def test_training_curve_aggregates_full_validation_interval() -> None:
    aggregate = aggregate_training_window(
        [
            {
                "cross_entropy": 2.0,
                "train_examples": 2,
                "microbatches": 2,
                "train_correct_by_modality": {"text": 1, "audio": 1},
                "train_examples_by_modality": {"text": 1, "audio": 1},
                "train_ce_by_modality": {"text": 3.0, "audio": 1.0},
            },
            {
                "cross_entropy": 1.0,
                "train_examples": 2,
                "microbatches": 2,
                "train_correct_by_modality": {"text": 1},
                "train_examples_by_modality": {"text": 2},
                "train_ce_by_modality": {"text": 1.0},
            },
        ]
    )
    assert aggregate["train_ce"] == pytest.approx(1.5)
    assert aggregate["train_accuracy"] == pytest.approx(0.75)
    assert aggregate["train_accuracy_by_modality"] == {
        "audio": 1.0,
        "text": pytest.approx(2 / 3),
    }
    assert aggregate["train_ce_by_modality"] == {"audio": 1.0, "text": 5 / 3}
    assert aggregate["training_examples_in_window"] == 4
    assert aggregate["microbatches_in_window"] == 4
    assert aggregate["optimizer_steps_in_window"] == 2


def test_runtime_projection_includes_validation_costs() -> None:
    projected = project_runtime_seconds(
        seconds_per_optimizer_step=2.0,
        optimizer_steps=512,
        evaluation_interval=128,
        mean_scheduled_evaluation_seconds=30.0,
        setup_evaluation_seconds=40.0,
        final_evaluation_seconds=20.0,
        other_overhead_seconds=10.0,
    )
    assert projected == 512 * 2 + 4 * 30 + 40 + 20 + 10


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


def test_teacher_v1_catalogs_are_pinned_and_training_sources_are_allow() -> None:
    from tiny_omni_decision.dataset import audit_manifest
    from tiny_omni_decision.schema import DatasetCatalog, DatasetManifest

    train_path = __import__("pathlib").Path("manifests/teacher-v1-training-corpus.yaml")
    heldout_path = __import__("pathlib").Path("manifests/teacher-v1-heldout-corpus.yaml")
    train_catalog = DatasetCatalog.model_validate(load_structured_file(train_path))
    heldout_catalog = DatasetCatalog.model_validate(load_structured_file(heldout_path))

    assert train_catalog.purpose == "training"
    assert heldout_catalog.purpose == "evaluation"
    for entry in train_catalog.sources:
        if not entry.include:
            continue
        source = DatasetManifest.model_validate(
            load_structured_file(train_path.parent / entry.manifest)
        )
        assert source.revision
        assert audit_manifest(source)["project_policy"] == "ALLOW"
    heldout_roles = {
        entry.manifest: entry.heldout_partition
        for entry in heldout_catalog.sources
        if entry.include
    }
    assert heldout_roles["candidates/open-jev-validation.yaml"] == "validation"
    assert heldout_roles["candidates/open-jev-test.yaml"] == "evaluation"
    assert heldout_roles["candidates/speech-commands-test.yaml"] == "evaluation"
    assert all(role is not None for role in heldout_roles.values())


def test_teacher_v1_sampling_policy_configs_share_rank16_and_seed17() -> None:
    configs = [
        "configs/decision/teacher_v1.yaml",
        "configs/decision/teacher_v1_weak_modalities.yaml",
        "configs/decision/teacher_v1_video_priority.yaml",
    ]

    for path in configs:
        config = decision_training_config(load_structured_file(path))
        assert config.seed == 17
        assert config.lora_rank == 16
        assert config.max_sample_repeats == 1
        assert set(config.modality_weights) == {"text", "image", "audio", "video"}


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


def test_video_task_weights_are_seeded_and_applied_within_the_video_bucket() -> None:
    weights = {
        "temporal_descriptive": 0.2,
        "explanatory": 0.3,
        "predictive": 0.3,
        "counterfactual": 0.2,
    }
    examples = [
        example(
            f"{task_type}-{index}",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri=f"source-ref://clevrer/video-{index}.mp4")],
            task_type=task_type,
        )
        for task_type in weights
        for index in range(100)
    ]

    first, source_counts = deterministic_sample_order(
        examples,
        seed=17,
        limit=20,
        modality_weights={"video": 1.0},
        video_task_weights=weights,
    )
    second, _ = deterministic_sample_order(
        examples,
        seed=17,
        limit=20,
        modality_weights={"video": 1.0},
        video_task_weights=weights,
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert Counter(item.task_type for item in first) == {
        "temporal_descriptive": 4,
        "explanatory": 6,
        "predictive": 6,
        "counterfactual": 4,
    }
    assert source_counts == {"video:MIT-IBM/CLEVRER": 20}
    assert len({item.id for item in first}) == 20


def test_video_task_sampling_uses_unique_parent_questions_before_sibling_choices() -> None:
    examples = [
        example(
            f"choice-{question}-{choice}",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri=f"source-ref://clevrer/video-{question}.mp4")],
            task_type="predictive",
            task_group_id=f"{question}:{question}",
        )
        for question in range(3)
        for choice in range(2)
    ]

    selected, _ = deterministic_sample_order(
        examples,
        seed=17,
        limit=3,
        modality_weights={"video": 1.0},
        video_task_weights={"predictive": 1.0},
    )

    assert len({item.task_group_id for item in selected}) == 3


def test_sampling_accounting_reports_unique_video_questions_by_task_type() -> None:
    examples = [
        example(
            "e-1",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri="source-ref://clevrer/video-1.mp4")],
            task_type="explanatory",
            task_group_id="1:7",
        ),
        example(
            "e-2",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri="source-ref://clevrer/video-1.mp4")],
            task_type="explanatory",
            task_group_id="1:7",
        ),
        example(
            "p-1",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri="source-ref://clevrer/video-2.mp4")],
            task_type="predictive",
            task_group_id="2:9",
        ),
    ]

    accounting = sampling_accounting(examples)

    assert accounting["consumed_by_task_type"] == {"explanatory": 2, "predictive": 1}
    assert accounting["unique_examples_by_task_type"] == {"explanatory": 2, "predictive": 1}
    assert accounting["unique_questions_by_task_type"] == {"explanatory": 1, "predictive": 1}


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


def test_sampling_normalizes_configured_modality_weights() -> None:
    examples = [
        *(example(f"t-{index}", "text-source") for index in range(300)),
        *(
            example(
                f"i-{index}",
                "image-source",
                "image",
                [MediaRef(kind="image", uri=f"image:{index}")],
            )
            for index in range(300)
        ),
        *(
            example(
                f"a-{index}",
                "audio-source",
                "audio",
                [MediaRef(kind="audio", uri=f"audio:{index}")],
            )
            for index in range(300)
        ),
        *(
            example(
                f"v-{index}",
                "video-source",
                "video",
                [MediaRef(kind="video", uri=f"video:{index}")],
            )
            for index in range(300)
        ),
    ]

    selected, _ = deterministic_sample_order(
        examples,
        seed=17,
        limit=600,
        modality_weights={"text": 1.5, "image": 1.5, "audio": 1.0, "video": 2.0},
    )

    modality_counts = {
        modality: sum(item.modality == modality for item in selected)
        for modality in ("text", "image", "audio", "video")
    }
    assert modality_counts == {"text": 150, "image": 150, "audio": 100, "video": 200}


def test_sampling_normalizes_source_weights_within_modality() -> None:
    examples = [
        *(example(f"a-{index}", "source-a") for index in range(100)),
        *(example(f"b-{index}", "source-b") for index in range(100)),
    ]

    _, counts = deterministic_sample_order(
        examples,
        seed=17,
        limit=90,
        modality_weights={"text": 1.0},
        source_weights={"source-a": 2.0, "source-b": 1.0},
    )

    assert counts == {"text:source-a": 60, "text:source-b": 30}


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


def test_training_config_parses_and_validates_video_task_weights() -> None:
    weights = {
        "temporal_descriptive": 0.2,
        "explanatory": 0.3,
        "predictive": 0.3,
        "counterfactual": 0.2,
    }

    config = decision_training_config({"sampling": {"video_task_weights": weights}})

    assert config.video_task_weights == weights
    with pytest.raises(ValueError, match="task weights"):
        DecisionTrainingConfig(video_task_weights={"predictive": -0.1})


def test_validation_reports_accuracy_and_nll_by_video_reasoning_type(monkeypatch) -> None:
    import torch

    trainer = available_helper("tiny_omni_decision.trainer", "_evaluate")
    trainer_module = importlib.import_module("tiny_omni_decision.trainer")
    examples = [
        example(
            "predictive-1",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri="source-ref://clevrer/video-1.mp4")],
            task_type="predictive",
        ),
        example(
            "counterfactual-1",
            "MIT-IBM/CLEVRER",
            "video",
            [MediaRef(kind="video", uri="source-ref://clevrer/video-2.mp4")],
            task_type="counterfactual",
        ),
    ]
    monkeypatch.setattr(
        trainer_module,
        "_forward_decision",
        lambda *args, **kwargs: (torch.tensor([0.0, 3.0]), 1),
    )

    class Model:
        def eval(self):
            return self

    metrics, predictions = trainer(
        Model(),
        None,
        examples,
        data_root=Path("."),
        config=DecisionTrainingConfig(),
    )

    assert metrics["video_type:predictive"]["count"] == 1
    assert metrics["video_type:predictive"]["accuracy"] == 1.0
    assert metrics["video_type:counterfactual"]["nll"] < 0.1
    assert [record["task_type"] for record in predictions] == [
        "predictive",
        "counterfactual",
    ]


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


def test_clevrer_questions_from_one_video_stay_together_deterministically() -> None:
    examples = [
        example(
            f"{scene}:{question}",
            "MIT-IBM/CLEVRER",
            "video",
            [
                MediaRef(
                    kind="video",
                    uri=f"source-ref://CLEVRER/revision/videos/validation/video_{scene}.mp4",
                )
            ],
        ).model_copy(
            update={
                "source_record_id": f"{scene}:{question}",
                "split": "validation",
                "question": f"Question {question} for scene {scene}",
            }
        )
        for scene in range(20)
        for question in range(5)
    ]

    validation, evaluation = partition_heldout_records(examples, seed=17)
    repeated_validation, repeated_evaluation = partition_heldout_records(examples, seed=17)

    assert [item.id for item in validation] == [item.id for item in repeated_validation]
    assert [item.id for item in evaluation] == [item.id for item in repeated_evaluation]
    validation_ids = {item.id for item in validation}
    split_by_video = {
        scene: {
            "validation" if item.id in validation_ids else "evaluation"
            for item in examples
            if item.source_record_id.startswith(f"{scene}:")
        }
        for scene in range(20)
    }
    assert all(len(roles) == 1 for roles in split_by_video.values())
    assert {next(iter(roles)) for roles in split_by_video.values()} == {"validation", "evaluation"}


def test_split_gate_rejects_reused_media_asset_with_different_questions() -> None:
    train = example(
        "train-image-question",
        "image-source",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/shared.png")],
    )
    evaluation = example(
        "eval-image-question",
        "image-source",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/shared.png")],
    ).model_copy(update={"state": "different state", "question": "different question"})

    with pytest.raises(ValueError, match="media identities"):
        check_train_eval_splits([train], [evaluation])


def test_split_gate_rejects_speech_commands_speaker_overlap() -> None:
    train = example(
        "train-audio",
        "google/speech_commands",
        "audio",
        [MediaRef(kind="audio", path="media/audio/train-clip.wav")],
    ).model_copy(update={"source_record_id": "yes/3f45de8a_nohash_0.wav", "split": "train"})
    validation = example(
        "validation-audio",
        "google/speech_commands",
        "audio",
        [MediaRef(kind="audio", path="media/audio/validation-clip.wav")],
    ).model_copy(
        update={
            "source_record_id": "no/3f45de8a_nohash_7.wav",
            "split": "validation",
        }
    )

    with pytest.raises(ValueError, match="source assets"):
        check_train_eval_splits([train], [validation])


def test_filter_previously_seen_records_removes_entire_shared_asset_groups() -> None:
    filter_seen = available_helper(
        "tiny_omni_decision.corpus", "filter_previously_seen_records"
    )
    used_image = example(
        "used-image-question",
        "sgvaze/clevr4",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/used.png")],
    )
    image_question = used_image.model_copy(
        update={"id": "new-image-question", "source_record_id": "new-image-question"}
    )
    same_speaker_used = example(
        "speaker-a-used",
        "google/speech_commands",
        "audio",
        [MediaRef(kind="audio", path="audio/speaker-a_nohash_0.wav")],
    ).model_copy(update={"source_record_id": "speaker-a_nohash_0.wav"})
    same_speaker_new = example(
        "speaker-a-new",
        "google/speech_commands",
        "audio",
        [MediaRef(kind="audio", path="audio/other-name.wav")],
    ).model_copy(update={"source_record_id": "speaker-a_nohash_7.wav"})
    unseen = example(
        "unseen-image",
        "sgvaze/clevr4",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/unseen.png")],
    )

    retained, report = filter_seen(
        [image_question, same_speaker_new, unseen], [used_image, same_speaker_used]
    )

    assert [item.id for item in retained] == ["unseen-image"]
    assert report["candidate_records"] == 3
    assert report["retained_records"] == 1
    assert report["excluded_records"] == 2
    assert report["excluded_by_reason"] == {"source_asset": 2}


def test_teacher_v1_corpus_freeze_excludes_legacy_audit_and_isolates_audit_file(
    tmp_path, monkeypatch
) -> None:
    from tiny_omni_decision import cli

    def split_examples(prefix: str, split: str) -> list[DecisionExample]:
        rows = []
        for modality in ("text", "image", "audio", "video"):
            media = []
            if modality == "image":
                media = [MediaRef(kind="image", uri=f"source-ref://{prefix}/image.png")]
            elif modality == "audio":
                media = [MediaRef(kind="audio", uri=f"source-ref://{prefix}/audio.wav")]
            elif modality == "video":
                media = [MediaRef(kind="video", uri=f"source-ref://{prefix}/video.mp4")]
            source = {
                "text": "TypeSafeAI/Open-Jev",
                "image": "sgvaze/clevr4",
                "audio": "google/speech_commands",
                "video": "MIT-IBM/CLEVRER",
            }[modality]
            rows.append(
                example(f"{prefix}-{modality}", source, modality, media).model_copy(
                    update={
                        "state": f"unique state {prefix} {modality}",
                        "split": split,
                    }
                )
            )
        return rows

    legacy_train = split_examples("legacy-train", "train")
    legacy_validation = split_examples("legacy-validation", "validation")
    legacy_evaluation = split_examples("legacy-evaluation", "test")
    legacy_dir = tmp_path / "v0"
    legacy_dir.mkdir()
    for name, rows in (
        ("train.jsonl", legacy_train),
        ("validation.jsonl", legacy_validation),
        ("eval.jsonl", legacy_evaluation),
    ):
        (legacy_dir / name).write_text(
            "".join(item.model_dump_json() + "\n" for item in rows), encoding="utf-8"
        )

    def fake_freeze_corpus(**kwargs) -> None:
        stage = kwargs["output_dir"]
        train = split_examples("new-train", "train")
        validation = split_examples("new-validation", "validation")
        audit = split_examples("new-audit", "test")
        audit.append(legacy_evaluation[0])
        for name, rows in (
            ("train.jsonl", train),
            ("validation.jsonl", validation),
            ("eval.jsonl", audit),
        ):
            (stage / name).write_text(
                "".join(item.model_dump_json() + "\n" for item in rows), encoding="utf-8"
            )
        (stage / "corpus-manifest.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": "a" * 64,
                    "content_fingerprint_algorithm": "test-fixture",
                    "train": {"sources": []},
                    "evaluation": {"sources": []},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(cli, "freeze_corpus", fake_freeze_corpus, raising=False)
    processed = tmp_path / "processed" / "durable-teacher-v1"
    sealed = tmp_path / "sealed" / "durable-teacher-v1"

    cli.freeze_teacher_v1_corpus(
        train_catalog_path=__import__("pathlib").Path(
            "manifests/teacher-v1-training-corpus.yaml"
        ),
        heldout_catalog_path=__import__("pathlib").Path(
            "manifests/teacher-v1-heldout-corpus.yaml"
        ),
        legacy_corpus_dir=legacy_dir,
        output_dir=processed,
        sealed_audit_dir=sealed,
        seed=17,
        max_records_per_source=8192,
    )

    train_rows = [
        DecisionExample.model_validate_json(line)
        for line in (processed / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audit_rows = [
        DecisionExample.model_validate_json(line)
        for line in (sealed / "sealed_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_rows) == 8
    assert len(audit_rows) == 4
    assert all(item.id != legacy_evaluation[0].id for item in audit_rows)
    assert not (processed / "sealed_audit.jsonl").exists()
    manifest = json.loads((processed / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert manifest["sealed_audit"]["audit_sha256"] == sha256_file(
        sealed / "sealed_audit.jsonl"
    )
    assert all(item["status"] == "disjoint" for item in manifest["pairwise_overlap"].values())


def test_teacher_v1_corpus_seed_is_immutable(tmp_path) -> None:
    from click.exceptions import ClickException

    from tiny_omni_decision import cli

    with pytest.raises(ClickException, match="seed is immutable at 17"):
        cli.freeze_teacher_v1_corpus(
            legacy_corpus_dir=tmp_path / "missing-v0",
            output_dir=tmp_path / "processed",
            sealed_audit_dir=tmp_path / "sealed",
            seed=19,
        )
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "sealed").exists()


def test_training_inputs_require_independent_validation_and_exclude_audit() -> None:
    validate = available_helper("tiny_omni_decision.corpus", "validate_training_inputs")

    with pytest.raises(ValueError, match="independent validation"):
        validate("train.jsonl", None)
    with pytest.raises(ValueError, match="same corpus"):
        validate("same.jsonl", "same.jsonl")
    with pytest.raises(ValueError, match="sealed audit"):
        validate("data/sealed-audit/train.jsonl", "data/validation.jsonl")
    with pytest.raises(ValueError, match="sealed audit"):
        validate("data/train.jsonl", "data/sealed/durable-teacher-v1/sealed_audit.jsonl")
    with pytest.raises(ValueError, match="evaluation is isolated"):
        validate("train.jsonl", "validation.jsonl", evaluation_path="audit.jsonl")


def test_run_training_fails_before_loading_data_when_validation_is_missing(tmp_path) -> None:
    from tiny_omni_decision.trainer import run_training

    with pytest.raises(ValueError, match="independent validation"):
        run_training(
            train_path=tmp_path / "train.jsonl",
            eval_path=tmp_path / "evaluation.jsonl",
            validation_path=None,
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "experiment",
        )


def test_metric_summary_reports_macro_and_weakest_modality() -> None:
    summarize = available_helper("tiny_omni_decision.corpus", "macro_metrics")

    actual = summarize(
        {
            "modality:text": {"accuracy": 0.5, "nll": 2.0, "brier": 0.4, "ece": 0.1},
            "modality:image": {"accuracy": 0.75, "nll": 1.0, "brier": 0.2, "ece": 0.2},
            "modality:audio": {"accuracy": 0.9, "nll": 0.5, "brier": 0.1, "ece": 0.05},
            "modality:video": {"accuracy": 0.25, "nll": 3.0, "brier": 0.6, "ece": 0.3},
            "all": {"accuracy": 0.7, "nll": 0.8, "brier": 0.2, "ece": 0.08},
        }
    )

    assert actual == {
        "macro_accuracy": pytest.approx(0.6),
        "minimum_modality_accuracy": pytest.approx(0.25),
        "macro_nll": pytest.approx(1.625),
        "macro_brier": pytest.approx(0.325),
        "macro_ece": pytest.approx(0.1625),
    }


def test_sampling_accounting_counts_unique_samples_assets_and_repeats() -> None:
    account = available_helper("tiny_omni_decision.training", "sampling_accounting")
    first = example("text-1", "text-source").model_copy(
        update={"source_record_id": "state-1", "state": "first state"}
    )
    second = example("text-2", "text-source").model_copy(
        update={"source_record_id": "state-2", "state": "second state"}
    )

    actual = account([first, second, first])

    assert actual["samples_consumed"] == 3
    assert actual["unique_examples"] == 2
    assert actual["repeated_example_count"] == 1
    assert actual["unique_underlying_assets"] == 2
    assert actual["unique_underlying_assets_by_source"] == {"text-source": 2}
    assert actual["unique_underlying_assets_by_source_modality"] == {
        "text-source:text": 2
    }
    assert actual["consumed_by_modality"] == {"text": 3}
    assert actual["unique_examples_by_source"] == {"text-source": 2}


def test_validation_selection_uses_equal_fixed_modality_counts() -> None:
    select = available_helper(
        "tiny_omni_decision.training", "deterministic_validation_subset"
    )
    examples = [
        *(example(f"t-{index}", "text-source") for index in range(12)),
        *(
            example(
                f"i-{index}",
                "image-source",
                "image",
                [MediaRef(kind="image", uri=f"source-ref://images/{index}.png")],
            )
            for index in range(3)
        ),
        *(
            example(
                f"a-{index}",
                "audio-source",
                "audio",
                [MediaRef(kind="audio", uri=f"source-ref://audio/{index}.wav")],
            )
            for index in range(9)
        ),
        *(
            example(
                f"v-{index}",
                "video-source",
                "video",
                [MediaRef(kind="video", uri=f"source-ref://video/{index}.mp4")],
            )
            for index in range(5)
        ),
    ]

    selected = select(examples, seed=17, limit=12)

    assert {
        modality: sum(item.modality == modality for item in selected)
        for modality in ("text", "image", "audio", "video")
    } == {
        "text": 3,
        "image": 3,
        "audio": 3,
        "video": 3,
    }


def test_corpus_statistics_reports_unique_examples_and_assets() -> None:
    first = example(
        "image-question-color",
        "image-source",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/one.png")],
    ).model_copy(update={"source_record_id": "image-1"})
    second = example(
        "image-question-shape",
        "image-source",
        "image",
        [MediaRef(kind="image", uri="source-ref://images/one.png")],
    ).model_copy(update={"source_record_id": "image-1"})

    stats = corpus_statistics([first, second])

    assert stats["unique_examples"] == 2
    assert stats["unique_underlying_assets"] == 1
    assert stats["unique_underlying_assets_by_modality"] == {"image": 1}
    assert stats["unique_underlying_assets_by_source"] == {"image-source": 1}
    assert stats["unique_underlying_assets_by_source_modality"] == {
        "image-source:image": 1
    }


def test_validation_checkpoint_selector_early_stops_on_plateau() -> None:
    selector_type = getattr(
        importlib.import_module("tiny_omni_decision.corpus"),
        "ValidationCheckpointSelector",
        None,
    )
    assert selector_type is not None, "missing validation-only checkpoint selector"
    selector = selector_type(patience=2, min_delta=0.0)
    good = {
        "modality:text": {"accuracy": 0.8, "nll": 0.5, "brier": 0.2, "ece": 0.1},
        "modality:image": {"accuracy": 0.7, "nll": 0.7, "brier": 0.3, "ece": 0.1},
        "modality:audio": {"accuracy": 0.9, "nll": 0.3, "brier": 0.1, "ece": 0.05},
        "modality:video": {"accuracy": 0.6, "nll": 0.9, "brier": 0.4, "ece": 0.15},
    }
    worse = {
        key: {**value, "nll": value["nll"] + 0.1}
        for key, value in good.items()
    }

    assert selector.observe(50, good) is True
    assert selector.observe(100, worse) is False
    assert selector.should_stop is False
    assert selector.observe(150, worse) is False
    assert selector.should_stop is True
    assert selector.best_step == 50


def test_experiment_manifest_round_trips_frozen_inputs_and_curves() -> None:
    manifest_type = getattr(
        importlib.import_module("tiny_omni_decision.experiment"),
        "ExperimentManifest",
        None,
    )
    assert manifest_type is not None, "missing serializable experiment manifest"
    manifest = manifest_type(
        experiment_id="seed17-rank16-512",
        status="completed",
        seed=17,
        base_model_repo_id="google/gemma-4-E2B-it-qat-q4_0-unquantized",
        base_model_revision="a" * 40,
        base_model_weights_sha256="b" * 64,
        train_corpus_sha256="a" * 64,
        validation_corpus_sha256="b" * 64,
        sealed_audit_corpus_sha256="c" * 64,
        config_sha256="c" * 64,
        sampling_policy={"modality_weights": {"text": 2, "image": 2, "audio": 1, "video": 2}},
        optimizer_schedule={"max_steps": 512, "gradient_accumulation_steps": 4},
        learning_curve=[{"step": 50, "train_ce": 1.2, "validation_nll": 1.3}],
        metrics={"validation": {"macro_accuracy": 0.7}},
        artifact={"best_step": 50, "trainable_parameter_count": 12345},
        environment={"gpu": "RTX 3080 Laptop"},
    )

    restored = manifest_type.model_validate_json(manifest.model_dump_json())

    assert restored == manifest


def test_experiment_ledger_keeps_started_and_failed_attempts(tmp_path) -> None:
    append_event = available_helper("tiny_omni_decision.experiment", "append_experiment_event")
    ledger = tmp_path / "experiments.jsonl"
    append_event(ledger, {"experiment_id": "failed-1", "status": "started"})
    append_event(ledger, {"experiment_id": "failed-1", "status": "failed", "failure": "oom"})

    events = [
        __import__("json").loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]

    assert [event["status"] for event in events] == ["started", "failed"]
    assert all(event["experiment_id"] == "failed-1" for event in events)


def test_sealed_audit_requires_frozen_selection_and_can_be_claimed_once(tmp_path) -> None:
    audit = importlib.import_module("tiny_omni_decision.sealed_audit")
    freeze = getattr(audit, "freeze_teacher_selection", None)
    claim_once = getattr(audit, "claim_sealed_audit_evaluation_once", None)
    assert callable(freeze) and callable(claim_once), "missing sealed-audit access guard"
    lock_path = tmp_path / "selection-lock.json"
    audit_manifest_path = tmp_path / "sealed-audit-manifest.json"
    audit_data_path = tmp_path / "sealed-audit.jsonl"
    claim_path = tmp_path / "sealed-audit-claim.json"

    with pytest.raises(ValueError, match="freeze candidate selection"):
        claim_once(lock_path, audit_manifest_path, audit_data_path, claim_path)
    assert not claim_path.exists()

    audit_data_path.write_text('{"id":"hidden"}\n', encoding="utf-8")
    actual_audit_hash = sha256_file(audit_data_path)
    audit_manifest_path.write_text(
        json.dumps({"sealed_audit_sha256": actual_audit_hash}), encoding="utf-8"
    )

    selection = {
        "teacher_id": "tiny-omni-decision-teacher-v1",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "train_corpus_sha256": "c" * 64,
        "validation_corpus_sha256": "d" * 64,
        "sealed_audit_sha256": actual_audit_hash,
        "sealed_audit_manifest_sha256": sha256_file(audit_manifest_path),
        "seed": 17,
        "best_step": 512,
        "selection_rule": "validation macro score v1",
        "sampling_policy": {"text": 2, "image": 2, "audio": 1, "video": 2},
    }
    freeze(lock_path, selection)

    claim = claim_once(lock_path, audit_manifest_path, audit_data_path, claim_path)

    assert claim["teacher_id"] == "tiny-omni-decision-teacher-v1"
    assert claim["sealed_audit_sha256"] == actual_audit_hash
    with pytest.raises(ValueError, match="already been attempted"):
        claim_once(lock_path, audit_manifest_path, audit_data_path, claim_path)


def test_sealed_audit_claim_checks_data_hash_before_consuming_single_attempt(tmp_path) -> None:
    from tiny_omni_decision.sealed_audit import claim_sealed_audit_evaluation_once

    lock_path = tmp_path / "selection-lock.json"
    manifest_path = tmp_path / "sealed-audit-manifest.json"
    data_path = tmp_path / "sealed-audit.jsonl"
    claim_path = tmp_path / "sealed-audit-claim.json"
    data_path.write_text('{"id":"actual"}\n', encoding="utf-8")
    freeze_teacher_selection = __import__(
        "tiny_omni_decision.sealed_audit", fromlist=["freeze_teacher_selection"]
    ).freeze_teacher_selection
    selection = {
        "teacher_id": "tiny-omni-decision-teacher-v1",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "train_corpus_sha256": "c" * 64,
        "validation_corpus_sha256": "d" * 64,
        "sealed_audit_sha256": "e" * 64,
        "sealed_audit_manifest_sha256": "f" * 64,
        "seed": 17,
        "best_step": 512,
        "selection_rule": "validation-only rule v1",
        "sampling_policy": {"text": 1, "image": 1, "audio": 1, "video": 1},
    }
    manifest_path.write_text(json.dumps({"sealed_audit_sha256": "e" * 64}), encoding="utf-8")
    selection["sealed_audit_manifest_sha256"] = sha256_file(manifest_path)
    freeze_teacher_selection(lock_path, selection)

    with pytest.raises(ValueError, match="data hash"):
        claim_sealed_audit_evaluation_once(lock_path, manifest_path, data_path, claim_path)

    assert not claim_path.exists()


def test_sealed_audit_cli_claims_before_loading_records(tmp_path, monkeypatch) -> None:
    import sys
    import types
    from pathlib import Path

    from tiny_omni_decision import cli, trainer
    from tiny_omni_decision.corpus import file_sha256
    from tiny_omni_decision.sealed_audit import artifact_sha256

    experiment_dir = tmp_path / "experiment"
    candidate = experiment_dir / "best"
    candidate.mkdir(parents=True)
    (candidate / "adapter_model.safetensors").write_bytes(b"adapter")
    metadata_path = experiment_dir / "run-metadata.json"
    manifest = load_structured_file("manifests/base-model.example.yaml")
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_role": "validation_selected_experiment_candidate",
                "best_adapter_path": "best",
                "best_adapter_sha256": file_sha256(
                    str(candidate / "adapter_model.safetensors")
                ),
                "base_repo_id": manifest["repo_id"],
                "base_revision": manifest["revision"],
                "best_checkpoint_step": 5,
            }
        ),
        encoding="utf-8",
    )
    config_path = Path("configs/decision/teacher_v1.yaml")
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("train\n", encoding="utf-8")
    validation_path.write_text("validation\n", encoding="utf-8")
    audit_dir = tmp_path / "data" / "sealed" / "durable-teacher-v1"
    audit_dir.mkdir(parents=True)
    audit_path = audit_dir / "sealed_audit.jsonl"
    audit_path.write_text("secret audit rows\n", encoding="utf-8")
    audit_hash = file_sha256(str(audit_path))
    audit_manifest_path = audit_dir / "sealed-audit-manifest.json"
    audit_manifest_path.write_text(
        json.dumps(
            {
                "sealed_audit_sha256": audit_hash,
                "sealed_audit_path": audit_path.name,
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "teacher_id": "tiny-omni-decision-teacher-v1",
                "checkpoint_sha256": artifact_sha256(candidate),
                "candidate_metadata_sha256": file_sha256(str(metadata_path)),
                "config_sha256": file_sha256(str(config_path)),
                "train_corpus_sha256": file_sha256(str(train_path)),
                "validation_corpus_sha256": file_sha256(str(validation_path)),
                "sealed_audit_sha256": audit_hash,
                "sealed_audit_manifest_sha256": file_sha256(str(audit_manifest_path)),
                "best_step": 5,
                "base_repo_id": manifest["repo_id"],
                "base_revision": manifest["revision"],
            }
        ),
        encoding="utf-8",
    )
    claim_path = audit_dir / "sealed-audit-claim.json"
    result_path = audit_dir / "sealed-audit-result.json"
    access_order: list[str] = []

    class FakeModel:
        def eval(self):
            return self

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path, is_trainable):
            return FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id, revision):
            return object()

    class FakeBaseModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForMultimodalLM=FakeBaseModel, AutoProcessor=FakeProcessor
        ),
    )

    def read_after_claim(path):
        assert claim_path.is_file()
        access_order.append("load-audit-records")
        return split_examples_for_cli_test()

    def evaluate_after_claim(*args, **kwargs):
        assert claim_path.is_file()
        access_order.append("evaluate")
        metrics = {
            f"modality:{name}": {
                "accuracy": 0.5,
                "nll": 1.0,
                "brier": 0.5,
                "ece": 0.1,
                "mean_confidence": 0.7,
            }
            for name in ("text", "image", "audio", "video")
        }
        metrics["all"] = {
            "accuracy": 0.5,
            "nll": 1.0,
            "brier": 0.5,
            "ece": 0.1,
            "mean_confidence": 0.7,
        }
        return metrics, []

    def split_examples_for_cli_test():
        rows = []
        for modality in ("text", "image", "audio", "video"):
            media = []
            if modality == "image":
                media = [MediaRef(kind="image", uri="source-ref://audit/image.png")]
            elif modality == "audio":
                media = [MediaRef(kind="audio", uri="source-ref://audit/audio.wav")]
            elif modality == "video":
                media = [MediaRef(kind="video", uri="source-ref://audit/video.mp4")]
            rows.append(
                example(f"audit-{modality}", f"source-{modality}", modality, media)
            )
        return rows

    monkeypatch.setattr(trainer, "_read_examples", read_after_claim)
    monkeypatch.setattr(trainer, "_evaluate", evaluate_after_claim)

    cli.evaluate_sealed_audit(
        candidate=candidate,
        config_path=config_path,
        train_path=train_path,
        validation_path=validation_path,
        audit_path=audit_path,
        audit_manifest_path=audit_manifest_path,
        lock_path=lock_path,
        claim_path=claim_path,
        result_path=result_path,
        model_manifest_path=Path("manifests/base-model.example.yaml"),
    )

    assert access_order == ["load-audit-records", "evaluate"]
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_metric_comparison_deltas() -> None:
    before = {
        "all": {
            "accuracy": 0.4,
            "nll": 2.0,
            "brier": 0.8,
            "ece": 0.3,
            "mean_confidence": 0.7,
        },
        "macro": {
            "macro_accuracy": 0.4,
            "minimum_modality_accuracy": 0.2,
            "macro_nll": 2.0,
            "macro_brier": 0.8,
            "macro_ece": 0.3,
        },
    }
    after = {
        "all": {
            "accuracy": 0.5,
            "nll": 1.5,
            "brier": 0.6,
            "ece": 0.2,
            "mean_confidence": 0.6,
        },
        "macro": {
            "macro_accuracy": 0.5,
            "minimum_modality_accuracy": 0.3,
            "macro_nll": 1.5,
            "macro_brier": 0.6,
            "macro_ece": 0.2,
        },
    }
    deltas = comparison_deltas(before, after)
    assert deltas["all"]["accuracy"] == pytest.approx(0.1)
    assert deltas["all"]["nll"] == pytest.approx(-0.5)
    assert deltas["all"]["mean_confidence"] == pytest.approx(-0.1)
    assert deltas["macro"]["macro_accuracy"] == pytest.approx(0.1)
    assert deltas["macro"]["minimum_modality_accuracy"] == pytest.approx(0.1)


def test_collation_metadata_and_eval_serialization() -> None:
    item = example("one", "source")
    metadata = collate_metadata(item, ["A", "B"])
    assert metadata["target_index"] == 1
    assert metadata["sample_id"] == item.id
    record = output_record(item, [1.5, 2.5], [0.2, 0.8])
    assert record["prediction"] == 1
    assert record["target"] == 1
    assert record["option_logits"] == [1.5, 2.5]


def test_best_adapter_publication_retries_transient_windows_rename_denial(
    tmp_path, monkeypatch
) -> None:
    from tiny_omni_decision.trainer import publish_best_adapter

    best = tmp_path / "best"
    best.mkdir()
    (best / "adapter_model.safetensors").write_bytes(b"previous-adapter")
    temporary = tmp_path / "best.tmp"
    temporary.mkdir()
    (temporary / "adapter_model.safetensors").write_bytes(b"selected-adapter")

    actual_replace = Path.replace
    failed_once = False

    def deny_first_promotion(source: Path, destination: Path) -> Path:
        nonlocal failed_once
        if source == temporary and destination == best and not failed_once:
            failed_once = True
            raise PermissionError("simulated transient Windows rename lock")
        return actual_replace(source, destination)

    monkeypatch.setattr(Path, "replace", deny_first_promotion)

    publish_best_adapter(temporary, best)

    assert failed_once is True
    assert (best / "adapter_model.safetensors").read_bytes() == b"selected-adapter"
    assert not temporary.exists()
    assert list(tmp_path.glob(".best-backup-*")) == []


def test_best_adapter_publication_restores_previous_adapter_after_retry_exhaustion(
    tmp_path, monkeypatch
) -> None:
    import tiny_omni_decision.trainer as trainer_module
    from tiny_omni_decision.trainer import publish_best_adapter

    best = tmp_path / "best"
    best.mkdir()
    (best / "adapter_model.safetensors").write_bytes(b"previous-adapter")
    temporary = tmp_path / "best.tmp"
    temporary.mkdir()
    (temporary / "adapter_model.safetensors").write_bytes(b"selected-adapter")

    actual_replace = Path.replace

    def deny_promotion(source: Path, destination: Path) -> Path:
        if source == temporary and destination == best:
            raise PermissionError("simulated persistent Windows rename lock")
        return actual_replace(source, destination)

    monkeypatch.setattr(Path, "replace", deny_promotion)
    monkeypatch.setattr(trainer_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="persistent Windows rename lock"):
        publish_best_adapter(temporary, best)

    assert (best / "adapter_model.safetensors").read_bytes() == b"previous-adapter"
    assert (temporary / "adapter_model.safetensors").read_bytes() == b"selected-adapter"
