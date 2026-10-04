from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from typer.testing import CliRunner

from tiny_omni_decision.cli import app
from tiny_omni_decision.corpus import partition_heldout_records, source_asset_identity
from tiny_omni_decision.dataset import (
    adapt_row,
    check_train_eval_splits,
    content_fingerprint,
    filter_mixed_license_rows,
    iter_hub_rows,
    iter_local_rows,
    normalize_jsonl,
    shuffle_options,
)
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import (
    DatasetCatalog,
    DatasetManifest,
    DecisionExample,
    LicenseProvenance,
    MediaRef,
)
from tiny_omni_decision.video_corpus import build_video_native_corpora

ROOT = Path(__file__).resolve().parents[1]


def manifest(name: str = "dataset.example.yaml") -> DatasetManifest:
    path = ROOT / "manifests" / name
    return DatasetManifest.model_validate(load_structured_file(path))


def test_open_jev_adapter_preserves_one_hot_source_target() -> None:
    source = manifest("candidates/open-jev.yaml")
    row = {
        "id": "case-1",
        "split": "train",
        "state_json": '"A test state"',
        "question": "Choose one",
        "options": ["red", "blue"],
        "target": [0.0, 1.0],
    }
    item = adapt_row(row, source, "open-jev")
    assert item.target == "blue"
    assert item.source_target == [0.0, 1.0]
    assert item.source_revision == source.revision


def test_open_jev_adapter_rejects_soft_target_instead_of_lossy_conversion() -> None:
    source = manifest("candidates/open-jev.yaml")
    row = {
        "id": "case-1",
        "state_json": '"s"',
        "question": "q",
        "options": ["a", "b"],
        "target": [0.7, 0.3],
    }
    with pytest.raises(ValueError, match="soft or malformed"):
        adapt_row(row, source, "open-jev")


def test_typed_decisions_synth_flattens_text_and_keeps_soft_teacher() -> None:
    item = manifest()
    case = {
        "state_id": "s1",
        "state": '"A synthetic case"',
        "state_is_json": True,
        "questions": json.dumps(
            {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route",
                    "criteria": {"a": "route A", "b": "route B"},
                },
                "safe": {
                    "type": "noul",
                    "instructions": "Is this safe?",
                    "criteria": {"true": "safe", "false": "unsafe"},
                },
            }
        ),
        "gold": json.dumps({"route": "a", "safe": True}),
        "teacher": json.dumps(
            {
                "route": {"probabilities": {"a": 0.9, "b": 0.1}},
                "safe": {"noul": 0.8},
            }
        ),
    }
    values = list(normalize_jsonl([case], item, "typed-decisions-synth", seed=3))
    assert len(values) == 2
    assert values[0].target == "a: route A"
    assert values[0].source_target["teacher"]["probabilities"]["a"] == 0.9
    assert values[1].target == "true"
    assert values[0].source_record_id.endswith(":route")


def test_clevr4_adapter_normalizes_image_taxonomies_with_source_rights() -> None:
    path = ROOT / "tests" / "fixtures" / "clevr4-annotations.json"
    raw = list(iter_local_rows(path))
    training_manifest = manifest("candidates/clevr4.yaml")
    evaluation_manifest = manifest("candidates/clevr4-validation.yaml")
    training_rows = [row for row in raw if row["split"] == "train"]
    evaluation_rows = [row for row in raw if row["split"] == "val"]
    training = list(normalize_jsonl(training_rows, training_manifest, "clevr4", seed=17))
    evaluation = list(normalize_jsonl(evaluation_rows, evaluation_manifest, "clevr4", seed=17))
    assert len(training) == len(evaluation) == 8
    assert {item.modality for item in training + evaluation} == {"image"}
    assert {item.question.rsplit(" ", 2)[-2].rstrip("?") for item in training} <= {
        "color",
        "texture",
        "count",
        "shape",
    }
    assert all(item.target in item.options and len(item.options) == 10 for item in training)
    assert all(item.provenance.license == "CC-BY-4.0" for item in training)
    assert all(
        item.source_target is not None and item.media[0].uri.startswith("source-ref://")
        for item in training
    )
    assert check_train_eval_splits(training, evaluation)["status"] == "disjoint"


def test_speech_commands_adapter_keeps_ten_words_and_audio_references_only() -> None:
    path = ROOT / "tests" / "fixtures" / "speech-commands.jsonl"
    rows = list(iter_local_rows(path))
    training = list(
        normalize_jsonl(rows, manifest("candidates/speech-commands.yaml"), "speech-commands")
    )
    evaluation = list(
        normalize_jsonl(rows, manifest("candidates/speech-commands-test.yaml"), "speech-commands")
    )
    assert len(training) == len(evaluation) == 2
    assert {item.modality for item in training + evaluation} == {"audio"}
    assert all(len(item.options) == 10 and item.target in item.options for item in training)
    assert all(item.provenance.license == "CC-BY-4.0" for item in training)
    assert all("/v0.02/" in item.media[0].uri for item in training + evaluation)
    assert "stop" in {item.target for item in evaluation}
    assert not any("_unknown_" in item.source_record_id for item in training)
    assert check_train_eval_splits(training, evaluation)["status"] == "disjoint"


def test_speech_commands_numeric_label_ids_follow_pinned_hub_class_order() -> None:
    source = manifest("candidates/speech-commands.yaml")
    rows = [
        {"id": "yes.wav", "split": "train", "label": 0},
        {"id": "stop.wav", "split": "train", "label": 8},
        {"id": "zero.wav", "split": "train", "label": 10},
        {"id": "silence.wav", "split": "train", "label": 35},
        {"id": "aux.wav", "split": "train", "label": 20},
        {"id": "unknown.wav", "split": "train", "label": 0, "is_unknown": True},
    ]
    items = list(normalize_jsonl(rows, source, "speech-commands"))
    assert {item.source_record_id: item.target for item in items} == {
        "yes.wav": "yes",
        "stop.wav": "stop",
    }


def test_clevrer_adapter_normalizes_cc0_video_descriptive_questions() -> None:
    path = ROOT / "tests" / "fixtures" / "clevrer-questions.json"
    rows = list(iter_local_rows(path))
    training = list(normalize_jsonl(rows, manifest("candidates/clevrer.yaml"), "clevrer"))
    evaluation = list(
        normalize_jsonl(rows, manifest("candidates/clevrer-validation.yaml"), "clevrer")
    )
    assert training and evaluation
    assert {item.modality for item in training + evaluation} == {"video"}
    assert all(item.provenance.license == "CC0-1.0" for item in training)
    assert all(item.target in item.options for item in training + evaluation)
    assert all("/videos/train/" in item.media[0].uri for item in training)
    assert all("/videos/validation/" in item.media[0].uri for item in evaluation)
    assert check_train_eval_splits(training, evaluation)["status"] == "disjoint"


def test_clevrer_video_native_adapter_flattens_reasoning_choices_and_tags_tasks() -> None:
    row = {
        "scene_index": 5,
        "video_filename": "video_00005.mp4",
        "questions": [
            {
                "question_id": 0,
                "question": "What color is the object?",
                "question_type": "descriptive",
                "question_subtype": "query_color",
                "answer": "red",
                "program": ["objects", "red", "query_color"],
            },
            {
                "question_id": 1,
                "question": "How many collisions occur before the cube enters?",
                "question_type": "descriptive",
                "question_subtype": "count",
                "answer": "2",
                "program": ["events", "filter_collision", "before", "count"],
            },
            {
                "question_id": 2,
                "question": "Which event explains the collision?",
                "question_type": "explanatory",
                "choices": [
                    {"choice_id": 0, "choice": "the blue sphere enters", "answer": "wrong"},
                    {"choice_id": 1, "choice": "the red cube moves", "answer": "correct"},
                ],
            },
            {
                "question_id": 3,
                "question": "What happens next?",
                "question_type": "predictive",
                "choices": [
                    {"choice_id": 0, "choice": "the cube collides", "answer": "correct"},
                    {"choice_id": 1, "choice": "the sphere exits", "answer": "wrong"},
                ],
            },
            {
                "question_id": 4,
                "question": "What would happen if the sphere were removed?",
                "question_type": "counterfactual",
                "choices": [
                    {"choice_id": 0, "choice": "the cube enters", "answer": "wrong"},
                    {"choice_id": 1, "choice": "no collision occurs", "answer": "correct"},
                ],
            },
        ],
    }
    source = manifest("candidates/clevrer.yaml")

    examples = list(normalize_jsonl([row], source, "clevrer-video-native"))

    assert [item.task_type for item in examples] == [
        "static_descriptive",
        "temporal_descriptive",
        "explanatory",
        "explanatory",
        "predictive",
        "predictive",
        "counterfactual",
        "counterfactual",
    ]
    assert examples[2].source_record_id == "5:2:0"
    assert examples[2].question == (
        "Which event explains the collision?\nCandidate statement: the blue sphere enters"
    )
    assert examples[2].options == ["wrong", "correct"]
    assert examples[2].target == "wrong"
    assert examples[3].target == "correct"
    assert examples[2].task_group_id == examples[3].task_group_id == "5:2"
    assert examples[2].task_group_id != examples[4].task_group_id
    assert len({source_asset_identity(item) for item in examples}) == 1
    assert len({item.id for item in examples}) == len(examples)

    legacy_examples = list(normalize_jsonl([row], source, "clevrer"))
    assert len(legacy_examples) == 2
    assert all(item.task_type is None for item in legacy_examples)


def test_clevrer_native_questions_and_choices_stay_grouped_by_video_deterministically() -> None:
    source = manifest("candidates/clevrer.yaml")
    rows = [
        {
            "scene_index": scene,
            "video_filename": f"video_{scene:05}.mp4",
            "questions": [
                {
                    "question_id": 7,
                    "question": "What happens next?",
                    "question_type": "predictive",
                    "choices": [
                        {"choice_id": 0, "choice": "a collision", "answer": "correct"},
                        {"choice_id": 1, "choice": "an exit", "answer": "wrong"},
                    ],
                },
                {
                    "question_id": 8,
                    "question": "What if the cube were removed?",
                    "question_type": "counterfactual",
                    "choices": [
                        {"choice_id": 0, "choice": "the sphere moves", "answer": "wrong"},
                        {"choice_id": 1, "choice": "nothing collides", "answer": "correct"},
                    ],
                },
            ],
        }
        for scene in (0, 1)
    ]
    examples = list(
        normalize_jsonl(rows, source, "clevrer-video-native")
    )

    first_train, first_validation = partition_heldout_records(
        examples, seed=17, validation_fraction=0.5
    )
    second_train, second_validation = partition_heldout_records(
        examples, seed=17, validation_fraction=0.5
    )

    def scene_membership(partition: list[DecisionExample]) -> dict[int, set[str]]:
        membership: dict[int, set[str]] = {}
        for item in partition:
            scene = int(item.source_record_id.split(":", 1)[0])
            membership.setdefault(scene, set()).add(item.source_record_id)
        return membership

    assert scene_membership(first_train) == scene_membership(second_train)
    assert scene_membership(first_validation) == scene_membership(second_validation)
    assert set(scene_membership(first_train)).isdisjoint(scene_membership(first_validation))
    assert len(scene_membership(first_train)) == len(scene_membership(first_validation)) == 1
    assert len(first_train) == len(first_validation) == 4
    assert check_train_eval_splits(first_train, first_validation)["status"] == "disjoint"


def test_video_native_corpus_replaces_only_video_rows_and_preserves_scene_splits() -> None:
    train_source = manifest("candidates/clevrer.yaml")
    validation_source = manifest("candidates/clevrer-validation.yaml")

    def video_row(scene: int) -> dict[str, object]:
        return {
            "scene_index": scene,
            "video_filename": f"video_{scene:05}.mp4",
            "questions": [
                {
                    "question_id": 0,
                    "question": "What color is the stationary object?",
                    "question_type": "descriptive",
                    "question_subtype": "query_color",
                    "answer": "red",
                    "program": ["objects", "unique", "query_color"],
                },
                {
                    "question_id": 1,
                    "question": "How many collisions happen before the sphere exits?",
                    "question_type": "descriptive",
                    "question_subtype": "count",
                    "answer": "1",
                    "program": ["filter_collision", "before", "filter_out", "count"],
                },
                {
                    "question_id": 2,
                    "question": "Which event explains the collision?",
                    "question_type": "explanatory",
                    "choices": [
                        {"choice_id": 0, "choice": "the sphere moves", "answer": "wrong"},
                        {"choice_id": 1, "choice": "the cube enters", "answer": "correct"},
                    ],
                },
                {
                    "question_id": 3,
                    "question": "What will happen next?",
                    "question_type": "predictive",
                    "choices": [
                        {"choice_id": 0, "choice": "the cube exits", "answer": "wrong"},
                        {"choice_id": 1, "choice": "the objects collide", "answer": "correct"},
                    ],
                },
                {
                    "question_id": 4,
                    "question": "What if the sphere were removed?",
                    "question_type": "counterfactual",
                    "choices": [
                        {"choice_id": 0, "choice": "the sphere enters", "answer": "wrong"},
                        {"choice_id": 1, "choice": "the cube remains", "answer": "correct"},
                    ],
                },
            ],
        }

    def text_example(split: str) -> DecisionExample:
        return DecisionExample(
            id=f"text-{split}",
            modality="text",
            state=f"State for {split}.",
            question="Which option?",
            options=["a", "b"],
            target="a",
            source="text-source",
            source_revision="a" * 40,
            source_record_id=split,
            split=split,
            provenance=LicenseProvenance(
                license="CC0-1.0",
                commercial_use=True,
                derivative_model_training_allowed=True,
                redistribution_allowed=True,
                trust_status="trusted",
            ),
        )

    frozen_train_video = normalize_jsonl(
        [video_row(0), video_row(1)], train_source, "clevrer"
    )
    train_media_by_scene = {
        scene: MediaRef(
            kind="video",
            path=f"raw/clevrer/videos/train/video_{scene:05}.mp4",
            sha256=f"{scene + 1:064x}",
            license="CC0-1.0",
        )
        for scene in (0, 1)
    }
    frozen_train_video = [
        item.model_copy(
            update={
                "media": [
                    train_media_by_scene[
                        int(item.source_record_id.split(":", 1)[0])
                    ]
                ]
            }
        )
        for item in frozen_train_video
    ]
    frozen_validation_video = normalize_jsonl(
        [video_row(10_000)], validation_source, "clevrer"
    )
    validation_media = MediaRef(
        kind="video",
        path="raw/clevrer/videos/validation/video_10000.mp4",
        sha256=f"{10_001:064x}",
        license="CC0-1.0",
    )
    frozen_validation_video = [
        item.model_copy(update={"media": [validation_media]})
        for item in frozen_validation_video
    ]
    base_train = [
        *frozen_train_video,
        text_example("train"),
    ]
    base_validation = [
        *frozen_validation_video,
        text_example("validation"),
    ]

    train, validation, report = build_video_native_corpora(
        base_train,
        base_validation,
        [video_row(0), video_row(1)],
        [video_row(10_000)],
        train_source,
        validation_source,
    )

    assert [item.id for item in train if item.modality == "text"] == ["text-train"]
    assert [item.id for item in validation if item.modality == "text"] == ["text-validation"]
    assert {item.task_type for item in train if item.modality == "video"} == {
        "temporal_descriptive",
        "explanatory",
        "predictive",
        "counterfactual",
    }
    assert {item.task_type for item in validation if item.modality == "video"} == {
        "temporal_descriptive",
        "explanatory",
        "predictive",
        "counterfactual",
    }
    assert report["train"]["unique_video_scenes"] == [0, 1]
    assert report["validation"]["unique_video_scenes"] == [10_000]
    assert report["split_integrity"]["shared_media_identities"] == 0
    assert report["split_integrity"]["shared_content_fingerprints"] == 0
    assert all(
        item.media == [train_media_by_scene[int(item.task_group_id.split(":", 1)[0])]]
        for item in train
        if item.modality == "video"
    )
    assert all(
        item.media == [validation_media] for item in validation if item.modality == "video"
    )
    assert check_train_eval_splits(train, validation)["status"] == "disjoint"


@pytest.mark.parametrize(
    ("adapter", "manifest_name", "row", "expected"),
    [
        (
            "onejev",
            "candidates/onejev.yaml",
            {
                "id": "img-1",
                "source": "gqa",
                "license": "cc-by-4.0",
                "split": "train",
                "state": "<image:1>",
                "question": "What?",
                "target": '{"a": 1.0, "b": 0.0}',
                "media": '[{"type":"image","path":"media/a.jpg"}]',
                "images": [],
            },
            "image",
        ),
        (
            "mmau",
            "candidates/mmau-test-mini.yaml",
            {
                "id": "audio-1",
                "audio_id": "clip.wav",
                "question": "Sound?",
                "choices": ["A", "B"],
                "answer": "B",
                "split": "test",
            },
            "audio",
        ),
        (
            "mvbench",
            "candidates/mvbench.yaml",
            {
                "video_id": "clip.mp4",
                "question": "Then?",
                "candidates": ["A", "B"],
                "answer": "A",
                "split": "test",
            },
            "video",
        ),
    ],
)
def test_multimodal_adapters_preserve_media_reference(
    adapter: str, manifest_name: str, row: dict[str, object], expected: str
) -> None:
    source = manifest(manifest_name)
    example = adapt_row(row, source, adapter)
    assert example.modality == expected
    assert example.media
    assert example.source_target is not None


def test_mixed_onejev_source_license_is_preserved() -> None:
    source = manifest("candidates/onejev.yaml")
    row = {
        "id": "row-1",
        "source": "ambiguous",
        "license": "none stated",
        "state": "s",
        "question": "Choose",
        "target": '{"yes": 1.0, "no": 0.0}',
        "media": None,
        "images": [],
    }
    item = adapt_row(row, source, "onejev", components={"ambiguous": {"license": "none stated"}})
    assert item.provenance.license == "none stated"
    assert item.provenance.source_component == "ambiguous"


def test_mixed_license_filter_keeps_only_affirmatively_allowed_components() -> None:
    rows = [
        {"id": "a", "source": "allowed", "license": "MIT"},
        {"id": "b", "source": "unknown", "license": "UNKNOWN"},
        {"id": "c", "source": "nc", "license": "CC-BY-NC-4.0"},
    ]
    components = {
        "allowed": {
            "license": "MIT",
            "commercial_use": True,
            "derivative_model_training_allowed": True,
            "redistribution_allowed": True,
            "trust_status": "trusted",
        },
        "unknown": {"license": "UNKNOWN", "trust_status": "review"},
        "nc": {
            "license": "CC-BY-NC-4.0",
            "commercial_use": False,
            "derivative_model_training_allowed": False,
            "redistribution_allowed": True,
            "trust_status": "trusted",
        },
    }
    accepted, counts = filter_mixed_license_rows(rows, components, has_media=False)
    assert [row["id"] for row in accepted] == ["a"]
    assert counts == {"ALLOW": 1, "REVIEW": 1, "DENY": 1, "unresolved_source": 0}


def test_mixed_license_filter_requires_media_rights_for_audio_rows() -> None:
    rows = [{"id": "audio-1", "source": "allowed", "license": "MIT", "audio_id": "clip.wav"}]
    components = {
        "allowed": {
            "license": "MIT",
            "commercial_use": True,
            "derivative_model_training_allowed": True,
            "redistribution_allowed": True,
            "trust_status": "trusted",
        }
    }
    accepted, counts = filter_mixed_license_rows(rows, components)
    assert accepted == []
    assert counts["REVIEW"] == 1


def test_deterministic_option_shuffle_preserves_target_and_source_answer() -> None:
    item = DecisionExample(
        id="x",
        modality="text",
        state="s",
        question="q",
        options=["a", "b", "c"],
        target="b",
        source="x",
        source_revision="a" * 40,
        source_record_id="1",
        split="train",
        source_target={"gold": "b", "teacher": [0.2, 0.7, 0.1]},
        provenance={"license": "MIT"},
    )
    one = shuffle_options(item, seed=42)
    two = shuffle_options(item, seed=42)
    assert one.options == two.options
    assert one.target == two.target == "b"
    assert one.source_target == item.source_target
    assert one.options != item.options


def test_option_order_duplicate_fingerprint_and_train_eval_overlap() -> None:
    train = DecisionExample(
        id="x1",
        modality="text",
        state=" Café  STATE ",
        question="Choose",
        options=["yes", "no"],
        target="yes",
        source="source",
        source_revision="a" * 40,
        source_record_id="id1",
        split="train",
        provenance={"license": "CC0-1.0"},
    )
    same_content = train.model_copy(
        update={
            "id": "x2",
            "options": ["no", "yes"],
            "target": "yes",
            "source_record_id": "id2",
            "split": "test",
        }
    )
    assert content_fingerprint(train) == content_fingerprint(same_content)
    with pytest.raises(ValueError, match="contamination"):
        check_train_eval_splits([train], [same_content])


def test_normalized_jsonl_serialization_roundtrip(tmp_path: Path) -> None:
    source = manifest()
    raw = {
        "state_id": "case",
        "state": '"hello"',
        "state_is_json": True,
        "questions": (
            '{"q":{"type":"noul","instructions":"Is it?","criteria":{"true":"yes","false":"no"}}}'
        ),
        "gold": '{"q":true}',
        "teacher": '{"q":{"noul":0.8}}',
    }
    item = next(normalize_jsonl([raw], source, "typed-decisions-synth"))
    path = tmp_path / "one.jsonl"
    path.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    loaded = DecisionExample.model_validate_json(path.read_text(encoding="utf-8").strip())
    assert loaded == item


def test_hub_jsonl_source_uses_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    source = manifest()
    source.notes["data_file"] = "data/train.jsonl"
    payload = b'{"state_id":"case-a"}\n{"state_id":"case-b"}\n'
    seen: list[str] = []

    def fake_open(url: str, timeout: int = 60) -> io.BytesIO:
        seen.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr("tiny_omni_decision.dataset.urllib.request.urlopen", fake_open)
    rows = iter_hub_rows(source)
    assert next(rows)["state_id"] == "case-a"
    rows.close()
    assert source.revision in seen[0]


def test_hub_gzip_jsonl_source_is_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    source = manifest("candidates/open-jev.yaml")
    source.notes["data_file"] = "raw/train.jsonl.gz"
    payload = gzip.compress(b'{"id":"row-1"}\n{"id":"row-2"}\n')

    def fake_open(url: str, timeout: int = 60) -> io.BytesIO:
        return io.BytesIO(payload)

    monkeypatch.setattr("tiny_omni_decision.dataset.urllib.request.urlopen", fake_open)
    rows = iter_hub_rows(source)
    assert next(rows)["id"] == "row-1"
    rows.close()


def test_hub_parquet_source_bypasses_legacy_dataset_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = manifest("candidates/speech-commands.yaml")
    source.notes["parquet_files"] = ["v0.02/train-00000-of-00006.parquet"]
    seen: dict[str, object] = {}

    class FakeRows(list[dict[str, object]]):
        features = {"audio": object()}

        def cast_column(self, name: str, feature: object) -> FakeRows:
            seen["cast_column"] = (name, feature)
            return self

    def fake_load_dataset(path: str, **kwargs: object) -> FakeRows:
        seen["path"] = path
        seen.update(kwargs)
        return FakeRows([{"file": "yes/example.wav", "label": 0}])

    fake_module = ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset
    fake_module.Audio = lambda decode: {"decode": decode}
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    rows = iter_hub_rows(source)
    assert next(rows)["label"] == 0
    assert seen == {
        "path": "parquet",
        "data_files": {
            "train": [
                "https://huggingface.co/datasets/google/speech_commands/resolve/"
                f"{source.revision}/v0.02/train-00000-of-00006.parquet"
            ]
        },
        "split": "train",
        "streaming": True,
        "cast_column": ("audio", {"decode": False}),
    }
    rows.close()


def test_duplicate_fixture_contains_unknown_and_multimodal_metadata() -> None:
    fixture = ROOT / "tests" / "fixtures" / "multimodal.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    items = [DecisionExample.model_validate(row) for row in rows]
    assert {item.modality for item in items} == {"text", "image", "audio", "video"}
    assert any(item.provenance.license == "UNKNOWN" for item in items)
    assert any(item.provenance.license == "CC-BY-NC-4.0" for item in items)
    training = [item for item in items if item.split == "train"]
    evaluation = [item for item in items if item.split == "test"]
    with pytest.raises(ValueError, match="contamination"):
        check_train_eval_splits(training, evaluation)


def test_invalid_target_fixture_is_rejected() -> None:
    path = ROOT / "tests" / "fixtures" / "invalid_target.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="target must match"):
        DecisionExample.model_validate(raw)


@pytest.mark.parametrize("catalog_name", ["training-candidates.yaml", "evaluation-candidates.yaml"])
def test_candidate_catalogs_are_separate_and_valid(catalog_name: str) -> None:
    path = ROOT / "manifests" / catalog_name
    catalog = DatasetCatalog.model_validate(load_structured_file(path))
    assert catalog.purpose == catalog_name.split("-")[0]
    assert any(source.include for source in catalog.sources)


def test_catalog_cli_accepts_candidates_for_each_declared_purpose() -> None:
    runner = CliRunner()
    training = runner.invoke(
        app, ["validate-dataset-catalog", str(ROOT / "manifests/training-candidates.yaml")]
    )
    evaluation = runner.invoke(
        app, ["validate-dataset-catalog", str(ROOT / "manifests/evaluation-candidates.yaml")]
    )
    assert training.exit_code == 0, training.output
    assert evaluation.exit_code == 0, evaluation.output


def test_training_catalog_rejects_noncommercial_evaluation_manifest(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    candidate_dir = manifest_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    mmau = manifest("candidates/mmau-test-mini.yaml")
    mmau.usage = "training"
    (candidate_dir / "mmau.yaml").write_text(
        yaml.safe_dump(mmau.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    catalog = {
        "schema_version": 1,
        "manifest_type": "candidate_catalog",
        "purpose": "training",
        "sources": [{"manifest": "candidates/mmau.yaml", "include": True, "adapter": "mmau"}],
    }
    catalog_path = manifest_dir / "training-candidates.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    result = CliRunner().invoke(app, ["validate-dataset-catalog", str(catalog_path)])
    assert result.exit_code != 0
    assert "DENY" in str(result.exception)


def test_catalogs_reject_same_split_even_when_dataset_revisions_differ(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifests"
    candidate_dir = manifest_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    training = manifest("candidates/speech-commands.yaml")
    training.split = "test"
    training.revision = "a" * 40
    evaluation = manifest("candidates/speech-commands-test.yaml")
    evaluation.revision = "b" * 40
    (candidate_dir / "speech-train.yaml").write_text(
        yaml.safe_dump(training.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    (candidate_dir / "speech-test.yaml").write_text(
        yaml.safe_dump(evaluation.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    training_catalog = {
        "schema_version": 1,
        "manifest_type": "candidate_catalog",
        "purpose": "training",
        "sources": [
            {
                "manifest": "candidates/speech-train.yaml",
                "include": True,
                "split": "test",
                "adapter": "speech-commands",
            }
        ],
    }
    evaluation_catalog = {
        "schema_version": 1,
        "manifest_type": "candidate_catalog",
        "purpose": "evaluation",
        "sources": [
            {
                "manifest": "candidates/speech-test.yaml",
                "include": True,
                "split": "test",
                "adapter": "speech-commands",
            }
        ],
    }
    (manifest_dir / "training-candidates.yaml").write_text(
        yaml.safe_dump(training_catalog), encoding="utf-8"
    )
    (manifest_dir / "evaluation-candidates.yaml").write_text(
        yaml.safe_dump(evaluation_catalog), encoding="utf-8"
    )
    result = CliRunner().invoke(
        app,
        ["validate-dataset-catalog", str(manifest_dir / "training-candidates.yaml")],
    )
    assert result.exit_code != 0
    assert "same dataset split" in str(result.exception)


def test_evaluation_catalog_rejects_train_split_even_without_catalog_split(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifests"
    candidate_dir = manifest_dir / "candidates"
    candidate_dir.mkdir(parents=True)
    source = manifest("candidates/speech-commands.yaml")
    source.usage = "evaluation"
    (candidate_dir / "speech.yaml").write_text(
        yaml.safe_dump(source.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    catalog_path = manifest_dir / "evaluation-candidates.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "manifest_type": "candidate_catalog",
                "purpose": "evaluation",
                "sources": [
                    {
                        "manifest": "candidates/speech.yaml",
                        "include": True,
                        "adapter": "speech-commands",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["validate-dataset-catalog", str(catalog_path)])
    assert result.exit_code != 0
    assert "training split" in str(result.exception)
