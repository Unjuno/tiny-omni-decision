from __future__ import annotations

import hashlib
import json
import math

import pytest


def _example(*, sample_id: str = "train-1", options: list[str] | None = None):
    from tiny_omni_decision.schema import DecisionExample

    values = options or ["red", "blue"]
    return DecisionExample.model_validate(
        {
            "id": sample_id,
            "modality": "text",
            "state": "",
            "question": "Which color is correct?",
            "options": values,
            "target": "blue",
            "source": "fixture/source",
            "source_revision": "a" * 40,
            "source_record_id": sample_id,
            "split": "train",
            "provenance": {"license": "CC0-1.0"},
        }
    )


def _cache_record(example, *, probabilities: list[float] | None = None) -> dict:
    probs = probabilities or [0.25, 0.75]
    return {
        "sample_id": example.id,
        "options": example.options,
        "target": example.target,
        "target_index": example.options.index(example.target),
        "option_order_sha256": hashlib.sha256(
            json.dumps(example.options, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "teacher_id": "frozen-teacher-v1",
        "teacher_revision": "teacher-revision-sha",
        "teacher_temperature": 1.0,
        "teacher_option_logits": [math.log(value) for value in probs],
        "teacher_option_probabilities": probs,
    }


def test_student_distillation_loss_matches_option_kl_ce_and_brier() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import student_option_distillation_loss

    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    teacher_probabilities = torch.tensor([0.25, 0.75])
    total, parts = student_option_distillation_loss(
        logits,
        teacher_probabilities,
        target_index=1,
        option_kl_weight=1.0,
        cross_entropy_weight=0.2,
        brier_weight=0.2,
    )
    expected_student = [1.0 / (1.0 + math.e), math.e / (1.0 + math.e)]
    expected_kl = sum(
        teacher * math.log(teacher / student)
        for teacher, student in zip([0.25, 0.75], expected_student, strict=True)
    )
    expected_ce = -math.log(expected_student[1])
    expected_brier = expected_student[0] ** 2 + (expected_student[1] - 1.0) ** 2
    assert float(total) == pytest.approx(expected_kl + 0.2 * expected_ce + 0.2 * expected_brier)
    assert float(parts["option_kl"]) == pytest.approx(expected_kl, abs=1e-7)
    assert float(parts["cross_entropy"]) == pytest.approx(expected_ce, abs=1e-7)
    assert float(parts["brier"]) == pytest.approx(expected_brier, abs=1e-7)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_teacher_cache_requires_exact_options_target_and_identity(tmp_path) -> None:
    from tiny_omni_decision.student_training import load_teacher_option_cache

    example = _example()
    path = tmp_path / "train-teacher-options.jsonl"
    path.write_text(json.dumps(_cache_record(example)) + "\n", encoding="utf-8")
    records = load_teacher_option_cache(
        path,
        [example],
        expected_teacher_id="frozen-teacher-v1",
        expected_teacher_revision="teacher-revision-sha",
        expected_temperature=1.0,
    )
    assert records[example.id]["teacher_option_probabilities"] == [0.25, 0.75]

    reordered = _cache_record(example, probabilities=[0.75, 0.25])
    reordered["options"] = list(reversed(example.options))
    reordered["target_index"] = 0
    path.write_text(json.dumps(reordered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="option order mismatch"):
        load_teacher_option_cache(
            path,
            [example],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )


def test_teacher_cache_rejects_missing_duplicate_and_wrong_teacher_records(tmp_path) -> None:
    from tiny_omni_decision.student_training import load_teacher_option_cache

    first = _example(sample_id="train-1")
    second = _example(sample_id="train-2")
    path = tmp_path / "train-teacher-options.jsonl"
    row = _cache_record(first)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing training sample"):
        load_teacher_option_cache(
            path,
            [first, second],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )

    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate teacher cache sample"):
        load_teacher_option_cache(
            path,
            [first],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )

    wrong_teacher = _cache_record(first)
    wrong_teacher["teacher_revision"] = "other-revision"
    path.write_text(json.dumps(wrong_teacher) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Teacher identity mismatch"):
        load_teacher_option_cache(
            path,
            [first],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )


def test_teacher_cache_is_train_only_and_rejects_boolean_probabilities(tmp_path) -> None:
    from tiny_omni_decision.student_training import load_teacher_option_cache

    example = _example()
    path = tmp_path / "train-teacher-options.jsonl"
    path.write_text(json.dumps(_cache_record(example)) + "\n", encoding="utf-8")
    validation_example = example.model_copy(update={"split": "validation"})
    with pytest.raises(ValueError, match="train-only"):
        load_teacher_option_cache(
            path,
            [validation_example],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )

    record = _cache_record(example)
    record["teacher_option_probabilities"] = [False, 1.0]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Teacher option values"):
        load_teacher_option_cache(
            path,
            [example],
            expected_teacher_id="frozen-teacher-v1",
            expected_teacher_revision="teacher-revision-sha",
            expected_temperature=1.0,
        )
