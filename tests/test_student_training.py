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
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_student_ternary_qat.py"
    emit_progress = runpy.run_path(str(script))["emit_progress"]

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


def test_explicit_latest_shadow_warm_start_checks_identity_and_step(tmp_path) -> None:
    import runpy
    from pathlib import Path

    helper = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "train_student_ternary_qat.py")
    )["verify_warm_start_ledger"]
    identity = {
        "student_revision": "pinned",
        "teacher_id": "teacher",
        "teacher_cache_sha256": "a" * 64,
        "train_id_order_sha256": "b" * 64,
        "validation_snapshot_sha256": "c" * 64,
        "train_corpus_sha256": "d" * 64,
        "target_element_count": 8,
        "target_parameter_names": ["0.weight"],
    }
    ledger = dict(
        identity,
        best_validation_step=128,
        global_step=256,
        latest_checkpoint_step=256,
        latest_shadow_filename="latest-shadow-step-0256.safetensors",
        latest_shadow_sha256="f" * 64,
    )
    # Even if best is 128, resume from the most recently saved step: 256.
    assert helper(
        ledger, identity, total_steps=1215, checkpoint_interval=128
    ) == 256
    bad = dict(ledger, teacher_cache_sha256="different")
    with pytest.raises(ValueError, match="teacher_cache_sha256"):
        helper(bad, identity, total_steps=1215, checkpoint_interval=128)
    with pytest.raises(ValueError, match="checkpoint interval"):
        helper(
            dict(
                ledger,
                latest_checkpoint_step=120,
                latest_shadow_filename="latest-shadow-step-0120.safetensors",
            ),
            identity, total_steps=1215, checkpoint_interval=128
        )
    with pytest.raises(ValueError, match="latest shadow filename"):
        helper(
            dict(ledger, latest_shadow_filename="best-shadow.safetensors"),
            identity, total_steps=1215, checkpoint_interval=128
        )
    # Legacy runs have no latest-shadow ledger. They may hand off the best
    # shadow only when that best is also the latest saved validation step.
    legacy_best_is_latest = dict(
        identity,
        global_step=384,
        examples_consumed=1536,
        best_validation_step=384,
        best_shadow_sha256="e" * 64,
    )
    assert helper(
        legacy_best_is_latest, identity, total_steps=1215, checkpoint_interval=128
    ) == 384
    legacy_best_is_stale = dict(
        legacy_best_is_latest, global_step=512, examples_consumed=2048
    )
    with pytest.raises(ValueError, match="best shadow is not the latest saved step"):
        helper(legacy_best_is_stale, identity, total_steps=1215, checkpoint_interval=128)
    with pytest.raises(ValueError, match="best shadow digest"):
        helper(
            dict(legacy_best_is_latest, best_shadow_sha256="bad"),
            identity,
            total_steps=1215,
            checkpoint_interval=128,
        )


def test_warm_start_shadow_restoration_validates_all_before_mutating() -> None:
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student_training import restore_qat_shadows
    from tiny_omni_decision.ternary import apply_ternary_qat

    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Linear(2, 2))
    targets = apply_ternary_qat(model, group_size=4)
    original = [model[i].parametrizations.weight.original.detach().clone() for i in (0, 1)]
    weights = {name: torch.ones_like(original[i]) for i, name in enumerate(targets)}
    restore_qat_shadows(model, targets, weights)
    for i in (0, 1):
        assert torch.equal(
            model[i].parametrizations.weight.original,
            torch.ones_like(original[i]),
        )
    # A later invalid tensor must not result in a partially applied restoration.
    before = model[0].parametrizations.weight.original.detach().clone()
    invalid = dict(weights)
    invalid[targets[-1]] = torch.ones(3, 3)
    with pytest.raises(ValueError, match="shape"):
        restore_qat_shadows(model, targets, invalid)
    assert torch.equal(model[0].parametrizations.weight.original, before)


def test_latest_shadow_publication_rotates_without_losing_last_checkpoint(tmp_path) -> None:
    import runpy
    from pathlib import Path

    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "train_student_ternary_qat.py")
    )
    publish = script["publish_latest_shadow"]
    data = {"global_step": 128, "best_validation_step": 128}
    metadata_path = tmp_path / "run-metadata.json"

    def save(path):
        path.write_bytes(f"weights for {path.name}".encode())

    publish(tmp_path, step=128, examples_seen=512, metadata=data, save_shadow=save)
    first = tmp_path / "latest-shadow-step-0128.safetensors"
    assert first.is_file()
    first_digest = data["latest_shadow_sha256"]
    assert data["latest_checkpoint_step"] == 128
    assert data["latest_checkpoint_examples_consumed"] == 512

    data["global_step"] = 256
    publish(tmp_path, step=256, examples_seen=1024, metadata=data, save_shadow=save)
    latest = tmp_path / "latest-shadow-step-0256.safetensors"
    assert latest.is_file()
    assert not first.exists()
    assert data["latest_checkpoint_step"] == 256
    assert data["latest_shadow_sha256"] != first_digest
    assert json.loads(metadata_path.read_text())["latest_shadow_filename"] == latest.name


def test_failed_latest_publication_keeps_prior_shadow_and_ledger(tmp_path) -> None:
    import runpy
    from pathlib import Path

    publish = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "train_student_ternary_qat.py")
    )["publish_latest_shadow"]
    data = {"global_step": 128}
    publish(
        tmp_path, step=128, examples_seen=512, metadata=data,
        save_shadow=lambda path: path.write_bytes(b"first"),
    )
    stored = dict(data)
    data["global_step"] = 256

    def failed_save(path):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        publish(
            tmp_path, step=256, examples_seen=1024,
            metadata=data, save_shadow=failed_save,
        )
    assert (tmp_path / stored["latest_shadow_filename"]).read_bytes() == b"first"
    record = json.loads((tmp_path / "run-metadata.json").read_text())
    assert record["latest_checkpoint_step"] == 128
    assert record["latest_shadow_sha256"] == stored["latest_shadow_sha256"]
