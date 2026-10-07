from __future__ import annotations

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
