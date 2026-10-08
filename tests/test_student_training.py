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


def test_ternary_qat_adafactor_uses_factored_state_and_updates_bf16_shadow() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import build_ternary_qat_optimizer

    shadow = torch.nn.Parameter(torch.ones((64, 128), dtype=torch.bfloat16))
    optimizer = build_ternary_qat_optimizer([shadow], name="adafactor", learning_rate=0.05)
    shadow.grad = torch.ones_like(shadow)
    optimizer.step()

    state = optimizer.state[shadow]
    assert set(state) == {"step", "row_var", "col_var"}
    assert state["row_var"].numel() + state["col_var"].numel() == 64 + 128
    assert state["row_var"].dtype == shadow.dtype
    assert torch.isfinite(shadow).all()
    assert not torch.equal(shadow, torch.ones_like(shadow))


def test_ternary_qat_optimizer_rejects_unknown_or_empty_configuration() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import build_ternary_qat_optimizer

    parameter = torch.nn.Parameter(torch.ones((2, 2)))
    with pytest.raises(ValueError, match="unsupported QAT optimizer"):
        build_ternary_qat_optimizer([parameter], name="unknown", learning_rate=1e-5)
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        build_ternary_qat_optimizer([], name="adafactor", learning_rate=1e-5)


def test_fp32_cpu_master_accumulates_updates_below_bf16_resolution() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import (
        build_ternary_qat_optimizer,
        copy_shadow_gradients_to_masters,
        make_fp32_cpu_master_parameters,
        sync_cpu_masters_to_shadows,
    )

    shadow = torch.nn.Parameter(torch.ones((64, 128), dtype=torch.bfloat16))
    masters = make_fp32_cpu_master_parameters([shadow])
    optimizer = build_ternary_qat_optimizer(masters, name="adafactor", learning_rate=0.001)

    for _ in range(2):
        shadow.grad = torch.ones_like(shadow)
        active = copy_shadow_gradients_to_masters([shadow], masters)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_cpu_masters_to_shadows([shadow], masters, active)
        shadow.grad = None

    assert masters[0].device.type == "cpu"
    assert masters[0].dtype == torch.float32
    assert not torch.equal(masters[0], torch.ones_like(masters[0]))
    assert not torch.equal(shadow, torch.ones_like(shadow))
    assert torch.isfinite(shadow).all()


def test_cpu_master_gradient_transfer_fails_closed_on_nonfinite_gradient() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import (
        copy_shadow_gradients_to_masters,
        make_fp32_cpu_master_parameters,
    )

    shadow = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.bfloat16))
    masters = make_fp32_cpu_master_parameters([shadow])
    shadow.grad = torch.full_like(shadow, float("inf"))
    with pytest.raises(ValueError, match="gradient must be finite"):
        copy_shadow_gradients_to_masters([shadow], masters)


def test_teacher_cache_requires_exact_options_target_and_identity(tmp_path) -> None:
    pytest.importorskip("torch")
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
    pytest.importorskip("torch")
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
    pytest.importorskip("torch")
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


def test_qat_script_progress_is_written_and_flushed(tmp_path, capsys) -> None:
    from scripts.train_student_ternary_qat import emit_progress

    output = tmp_path / "latest-progress.json"
    emit_progress(
        output,
        event="train_update",
        step=16,
        examples_seen=64,
        elapsed_seconds=42.5,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "event": "train_update",
        "step": 16,
        "examples_seen": 64,
        "elapsed_seconds": 42.5,
    }
    assert json.loads(capsys.readouterr().out.strip()) == payload
