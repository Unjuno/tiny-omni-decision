from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _example(*, sample_id: str, options: list[str], target: str):
    from tiny_omni_decision.schema import DecisionExample

    return DecisionExample.model_validate(
        {
            "id": sample_id,
            "modality": "text",
            "state": "",
            "question": "Pick the matching answer.",
            "options": options,
            "target": target,
            "source": "fixture/source",
            "source_revision": "a" * 40,
            "source_record_id": sample_id,
            "split": "validation",
            "provenance": {"license": "CC0-1.0"},
        }
    )


def test_student_eval_preserves_example_and_option_order(monkeypatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.student_eval as student_eval

    example = _example(sample_id="sample-1", options=["wrong", "right"], target="right")

    class RecordingProcessor:
        def __call__(self, *, text, return_tensors):
            return {"text": text}

    def embeddings(_model, inputs):
        text = inputs["text"]
        if len(text) == 1:
            return torch.tensor([[1.0, 0.0]])
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    monkeypatch.setattr(student_eval, "model_sentence_embeddings", embeddings)
    model = SimpleNamespace(eval=lambda: None)

    metrics, predictions = student_eval.evaluate_student_examples(
        model,
        RecordingProcessor(),
        [example],
        data_root=tmp_path,
        temperature=1.0,
    )

    prediction = predictions[0]
    assert prediction["sample_id"] == "sample-1"
    assert prediction["options"] == ["wrong", "right"]
    assert prediction["target"] == 1
    assert prediction["prediction"] == 1
    assert len(prediction["option_order_sha256"]) == 64
    assert metrics["all"]["count"] == 1
    assert metrics["modality:text"]["accuracy"] == 1.0
    assert metrics["source:fixture/source"]["nll"] == pytest.approx(0.3132617)
    assert metrics["macro_modality"]["count"] == 1
    assert metrics["minimum_modality_accuracy"] == 1.0


def test_student_eval_rejects_duplicate_ids_and_invalid_temperature(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from tiny_omni_decision.student_eval import evaluate_student_examples

    example = _example(sample_id="duplicate", options=["a", "b"], target="a")
    model = SimpleNamespace(eval=lambda: None)

    with pytest.raises(ValueError, match="temperature"):
        evaluate_student_examples(
            model, object(), [example], data_root=tmp_path, temperature=0.0
        )
    with pytest.raises(ValueError, match="IDs must be unique"):
        evaluate_student_examples(
            model, object(), [example, example], data_root=tmp_path, temperature=1.0
        )


def test_student_eval_reuses_option_embeddings_within_one_run(
    monkeypatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.student_eval as student_eval

    examples = [
        _example(sample_id="first", options=["wrong", "right"], target="right"),
        _example(sample_id="second", options=["right", "other"], target="right"),
    ]

    class RecordingProcessor:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *, text, return_tensors):
            self.calls.append(text)
            return {"text": text}

    def embeddings(_model, inputs):
        texts = inputs["text"]
        if "Pick the matching answer." in texts[0]:
            return torch.tensor([[1.0, 0.0]])
        vectors = {
            "wrong": [0.0, 1.0],
            "right": [1.0, 0.0],
            "other": [-1.0, 0.0],
        }
        return torch.tensor([vectors[text.rsplit(" ", 1)[-1]] for text in texts])

    monkeypatch.setattr(student_eval, "model_sentence_embeddings", embeddings)
    processor = RecordingProcessor()
    model = SimpleNamespace(eval=lambda: None)

    _, predictions = student_eval.evaluate_student_examples(
        model, processor, examples, data_root=tmp_path, temperature=1.0
    )

    assert [len(call) for call in processor.calls] == [1, 2, 1, 1]
    assert [row["prediction"] for row in predictions] == [1, 0]
    assert [row["options"] for row in predictions] == [
        ["wrong", "right"],
        ["right", "other"],
    ]


def test_video_frame_cache_decodes_a_scene_once_and_passes_original_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    import sys
    import types

    from tiny_omni_decision.schema import DecisionExample
    from tiny_omni_decision.student import processor_inputs_for_decision_example

    video_path = tmp_path / "scene.mp4"
    video_path.write_bytes(b"fixture")
    example = DecisionExample.model_validate(
        {
            "id": "video-one",
            "modality": "video",
            "state": "",
            "question": "What happens?",
            "options": ["a", "b"],
            "target": "a",
            "media": [{"kind": "video", "path": "scene.mp4"}],
            "source": "fixture/source",
            "source_revision": "a" * 40,
            "source_record_id": "video-one",
            "split": "validation",
            "provenance": {"license": "CC0-1.0"},
        }
    )
    decoded = (["frame-0", "frame-1"], {"fps": 25, "total_num_frames": 50})
    decoder_calls = []

    def load_video(path, *, backend):
        decoder_calls.append((path, backend))
        return decoded

    transformers = types.ModuleType("transformers")
    video_utils = types.ModuleType("transformers.video_utils")
    video_utils.load_video = load_video
    transformers.video_utils = video_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.video_utils", video_utils)

    class RecordingProcessor:
        video_token = "<video>"

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs

    cache = {}
    processor = RecordingProcessor()
    first = processor_inputs_for_decision_example(
        processor, example, data_root=tmp_path, video_frame_cache=cache
    )
    second = processor_inputs_for_decision_example(
        processor, example, data_root=tmp_path, video_frame_cache=cache
    )

    assert decoder_calls == [(str(video_path.resolve()), "pyav")]
    assert first == second
    assert first["videos"] == [decoded[0]]
    assert first["videos_kwargs"]["video_metadata"] == decoded[1]


def test_fixed_validation_loader_checks_hash_ids_targets_and_option_order(
    tmp_path: Path,
) -> None:
    from tiny_omni_decision.student_eval import load_fixed_validation_snapshot

    examples = [
        _example(sample_id="sample-1", options=["a", "b"], target="b"),
        _example(sample_id="sample-2", options=["b", "a"], target="a"),
    ]
    rows = [json.loads(example.model_dump_json()) for example in examples]
    source_path = tmp_path / "source-validation.jsonl"
    snapshot_path = tmp_path / "snapshot.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    snapshot_path.write_bytes(source_path.read_bytes())
    teacher_rows = [
        {
            "sample_id": example.id,
            "source": example.source,
            "modality": example.modality,
            "target": example.options.index(example.target),
            "option_probabilities": [0.2, 0.8],
        }
        for example in examples
    ]
    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in teacher_rows),
        encoding="utf-8",
    )
    option_hashes = {
        example.id: hashlib.sha256(
            json.dumps(example.options, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        for example in examples
    }
    manifest = {
        "split": "validation",
        "selected_validation_jsonl_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "source_validation_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "teacher_best_validation_predictions_sha256": hashlib.sha256(
            teacher_path.read_bytes()
        ).hexdigest(),
        "sample_ids": [example.id for example in examples],
        "record_count": len(examples),
        "option_order_sha256": option_hashes,
        "modality_counts": {"text": 2},
        "source_counts": {"fixture/source": 2},
    }
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, loaded_teacher = load_fixed_validation_snapshot(
        snapshot_path,
        manifest_path,
        source_validation_path=source_path,
        teacher_predictions_path=teacher_path,
    )

    assert [example.id for example in loaded] == ["sample-1", "sample-2"]
    assert [row["sample_id"] for row in loaded_teacher] == ["sample-1", "sample-2"]


def test_paired_comparison_rejects_different_sample_order() -> None:
    from tiny_omni_decision.student_eval import compare_student_predictions

    student = [
        {
            "sample_id": "first",
            "source": "s",
            "modality": "text",
            "target": 0,
            "options": ["a", "b"],
            "option_probabilities": [0.8, 0.2],
        }
    ]
    teacher = [
        {
            "sample_id": "second",
            "source": "s",
            "modality": "text",
            "target": 0,
            "option_probabilities": [0.7, 0.3],
        }
    ]

    with pytest.raises(ValueError, match="not paired"):
        compare_student_predictions(student, teacher)


def test_local_checkpoint_file_verifier_checks_pinned_hash_and_size(tmp_path: Path) -> None:
    import hashlib

    from tiny_omni_decision.schema import BaseModelManifest
    from tiny_omni_decision.student_evaluate import _verify_local_model_files

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pinned weights")
    manifest = BaseModelManifest.model_validate(
        {
            "repo_id": "fixture/model",
            "revision": "a" * 40,
            "role": "test",
            "license": "Apache-2.0",
            "modalities": ["text"],
            "processor_revision": "a" * 40,
            "files": [
                {
                    "path": "model.safetensors",
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "size_bytes": weights.stat().st_size,
                }
            ],
        }
    )

    assert _verify_local_model_files(tmp_path, manifest) == {
        "model.safetensors": hashlib.sha256(weights.read_bytes()).hexdigest()
    }
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_local_model_files(tmp_path, manifest)


def test_media_preflight_requires_safe_existing_local_assets(tmp_path: Path) -> None:
    from tiny_omni_decision.schema import MediaRef
    from tiny_omni_decision.student_eval import validate_local_media_paths

    text_example = _example(sample_id="text", options=["a", "b"], target="a")
    image_example = text_example.model_copy(
        update={
            "id": "image",
            "modality": "image",
            "media": [MediaRef(kind="image", path="images/sample.png")],
        }
    )
    image = tmp_path / "images" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"fixture")

    assert validate_local_media_paths([text_example, image_example], tmp_path) == {"image": 1}
    image.unlink()
    with pytest.raises(FileNotFoundError, match="media file is missing"):
        validate_local_media_paths([image_example], tmp_path)


def test_student_eval_resumes_only_from_an_exact_prediction_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    import tiny_omni_decision.student_eval as student_eval

    examples = [
        _example(sample_id="first", options=["wrong", "right"], target="right"),
        _example(sample_id="second", options=["right", "other"], target="right"),
    ]

    class RecordingProcessor:
        def __call__(self, *, text, return_tensors):
            return {"text": text}

    def embeddings(_model, inputs):
        texts = inputs["text"]
        if "Pick the matching answer." in texts[0]:
            return torch.tensor([[1.0, 0.0]])
        vectors = {"wrong": [0.0, 1.0], "right": [1.0, 0.0], "other": [-1.0, 0.0]}
        return torch.tensor([vectors[text.rsplit(" ", 1)[-1]] for text in texts])

    monkeypatch.setattr(student_eval, "model_sentence_embeddings", embeddings)
    model = SimpleNamespace(eval=lambda: None)
    first_metrics, first_predictions = student_eval.evaluate_student_examples(
        model,
        RecordingProcessor(),
        examples[:1],
        data_root=tmp_path,
        temperature=1.0,
    )
    callbacks = []
    metrics, predictions = student_eval.evaluate_student_examples(
        model,
        RecordingProcessor(),
        examples,
        data_root=tmp_path,
        temperature=1.0,
        existing_predictions=first_predictions,
        on_prediction=callbacks.append,
    )

    assert first_metrics["all"]["count"] == 1
    assert metrics["all"]["count"] == 2
    assert [row["sample_id"] for row in predictions] == ["first", "second"]
    assert [row["sample_id"] for row in callbacks] == ["second"]

    reordered = [dict(first_predictions[0], sample_id="second")]
    with pytest.raises(ValueError, match="exact validation prefix"):
        student_eval.evaluate_student_examples(
            model,
            RecordingProcessor(),
            examples,
            data_root=tmp_path,
            temperature=1.0,
            existing_predictions=reordered,
        )


def test_student_evaluation_applies_overlay_for_its_pinned_base(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_evaluate import apply_student_ternary_overlay
    from tiny_omni_decision.ternary import export_packed_ternary_overlay

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 1, bias=False)

    source = TinyModel()
    with torch.no_grad():
        source.projection.weight.copy_(torch.tensor([[-3.0, -1.0, 1.0, 3.0]]))
    overlay = tmp_path / "packed-overlay"
    export_packed_ternary_overlay(
        source,
        overlay,
        base_model_id="google/embeddinggemma-2",
        base_revision="revision-123",
        group_size=4,
    )
    loaded = TinyModel()
    with torch.no_grad():
        loaded.projection.weight.zero_()

    apply_student_ternary_overlay(
        loaded,
        overlay,
        expected_base_model_id="google/embeddinggemma-2",
        expected_base_revision="revision-123",
    )

    torch.testing.assert_close(
        loaded.projection.weight, torch.tensor([[-3.0, 0.0, 0.0, 3.0]])
    )
