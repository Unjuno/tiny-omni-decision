import importlib.util
import json

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def test_recovery_module_exists():
    assert importlib.util.find_spec("tiny_omni_decision.student.recovery") is not None


def model():
    return torch.nn.Sequential(torch.nn.Linear(4, 3))


@pytest.mark.parametrize("lora", [False, True])
def test_packed_export_reload_preserves_predictions(tmp_path, lora):
    from tiny_omni_decision.student.artifacts import load_bundle, save_bundle
    from tiny_omni_decision.student.ternary import TernaryController
    torch.manual_seed(7)
    m = model()
    c = TernaryController(m, group_size=5)
    if lora:
        c.freeze()
        c.add_lora(rank=2, alpha=4)
        with torch.no_grad():
            c.loras["0.weight"].lora_B.fill_(0.25)
    x = torch.randn(5, 4)
    expected = m(x).detach()
    destination = tmp_path / "bundle"
    report = save_bundle(c, destination, {"stage": "lora" if lora else "qat"})
    assert report["total_bytes"] == sum(p.stat().st_size for p in destination.iterdir())
    assert report["adapter_bytes"] > 0 if lora else report["adapter_bytes"] == 0
    from safetensors.torch import load_file
    assert not any("original" in n or "shadow" in n for n in load_file(
        destination / "model.safetensors"
    ))
    fresh = model()
    fresh_c = TernaryController(fresh, group_size=5)
    load_bundle(fresh_c, destination)
    assert torch.equal(expected, fresh(x))
    assert all(not p.requires_grad for p in fresh.parameters())
    with pytest.raises(FileExistsError):
        save_bundle(c, destination, {})


def test_export_integrity_check(tmp_path):
    from tiny_omni_decision.student.artifacts import load_bundle, save_bundle
    from tiny_omni_decision.student.ternary import TernaryController
    c = TernaryController(model(), group_size=4)
    destination = tmp_path / "bundle"
    save_bundle(c, destination, {})
    f = destination / "model.safetensors"
    f.write_bytes(f.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(TernaryController(model(), group_size=4), destination)


def record(identifier="1", options=None, logits=None):
    from tiny_omni_decision.student.data import Record
    return Record(
        sample_id=identifier, group_id="g" + identifier, content_id="c" + identifier,
        modality="text", inputs={"text": "question " + identifier},
        options=options or ["one", "two", "three"], target=1,
        teacher_logits=logits or [-1.0, 2.0, 0.0], media_hashes={},
    )


def test_cache_is_role_and_option_bound(tmp_path):
    from tiny_omni_decision.student.data import load_cache, write_cache
    r = record()
    path = tmp_path / "train.jsonl"
    write_cache(path, [r], role="train", teacher_id="teacher-sha", source_sha256="a" * 64)
    cache = load_cache(path, "train")
    assert cache.records[0].options == ["one", "two", "three"]
    with pytest.raises(ValueError, match="role"):
        load_cache(path, "validation")
    with pytest.raises(FileExistsError):
        write_cache(path, [r], role="train", teacher_id="t", source_sha256="a" * 64)
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["options"] = ["two", "one", "three"]
    path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="digest"):
        load_cache(path, "train")


@pytest.mark.parametrize("change", [
    {"options": ["same", "same", "other"]},
    {"teacher_logits": [1.0, 2.0]},
    {"teacher_logits": [1.0, float("nan"), 2.0]},
    {"target": True},
    {"target": 99},
])
def test_invalid_record_rejected(change):
    from dataclasses import asdict

    from tiny_omni_decision.student.data import Record
    values = asdict(record()) | change
    with pytest.raises(ValueError):
        Record(**values)


def test_split_contamination_and_teacher_change_rejected():
    from tiny_omni_decision.student.data import Cache, assert_disjoint
    a = Cache("train", "teacher", [record("1")], "a" * 64)
    b = Cache("validation", "teacher", [record("1")], "b" * 64)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(a, b)
    b = Cache("validation", "different-teacher", [record("2")], "b" * 64)
    with pytest.raises(ValueError, match="Teacher"):
        assert_disjoint(a, b)


def test_distillation_uses_detached_teacher_and_variable_option_count():
    from tiny_omni_decision.student.recovery import distillation_loss
    s = torch.tensor([0.2, 0.5, -0.3], requires_grad=True)
    t = torch.tensor([0.2, 0.5, -0.3], requires_grad=True)
    kd_only = distillation_loss(s, t, 1, temperature=2, ce_weight=0, brier_weight=0)
    assert abs(kd_only.item()) < 1e-6
    loss = distillation_loss(s, t + torch.tensor([2.0, 0.0, 0.0]), 1)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None
    with pytest.raises(ValueError):
        distillation_loss(s, t[:2], 1)
    with pytest.raises(ValueError):
        distillation_loss(s, t, 1, temperature=0)


def test_per_modality_gate_cannot_hide_bad_modality():
    from tiny_omni_decision.student.recovery import quality_gate
    teacher = {"all": {"accuracy": 0.8, "nll": 0.5, "brier": 0.2, "ece": 0.1},
               "audio": {"accuracy": 0.9, "nll": 0.3, "brier": 0.1, "ece": 0.05}}
    student = {"all": dict(teacher["all"]), "audio": dict(teacher["audio"], accuracy=0.2)}
    limits = dict(max_accuracy_drop=0.05, max_nll_increase=0.1,
                  max_brier_increase=0.05, max_ece_increase=0.03)
    assert not quality_gate(student, teacher, limits)["passed"]
    assert quality_gate(teacher, teacher, limits)["passed"]


def test_recovery_selects_and_reloads_then_uses_lora_only_when_needed(tmp_path):
    from tiny_omni_decision.student.data import Cache
    from tiny_omni_decision.student.recovery import run_recovery
    from tiny_omni_decision.student.ternary import TernaryController
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
        def forward(self, item):
            return self.linear(torch.tensor([1.0, -0.5, 0.25, 2.0]))
    limits = dict(max_accuracy_drop=1.0, max_nll_increase=100.0,
                  max_brier_increase=10.0, max_ece_increase=1.0)
    train = Cache("train", "t", [record("1"), record("2")], "a" * 64)
    val = Cache("validation", "t", [record("3")], "b" * 64)
    config = dict(qat_steps=3, lora_steps=2, evaluation_interval=1,
                  learning_rate=0.01, lora_learning_rate=0.01, seed=17,
                  lora_rank=2, lora_alpha=4.0, limits=limits)
    torch.manual_seed(12)
    m = Toy()
    c = TernaryController(m, group_size=4)
    result = run_recovery(m, c, train, val, tmp_path / "pass", config, metadata={})
    assert result["status"] == "PASS"
    assert not result["lora_used"]
    assert not c.loras
    assert result["reload_max_abs_error"] == 0.0
    strict = dict(max_accuracy_drop=0.0, max_nll_increase=0.0,
                  max_brier_increase=0.0, max_ece_increase=0.0)
    # A very confident teacher makes a deliberately unmet small-budget gate.
    difficult_val = Cache("validation", "t", [record("3", logits=[-100, 100, -100])], "b"*64)
    m = Toy()
    c = TernaryController(m, group_size=4)
    result = run_recovery(m, c, train, difficult_val, tmp_path / "fail",
                          config | {"limits": strict}, metadata={})
    assert result["lora_used"]
    assert c.loras and all(q.frozen for q in c.targets.values())
    assert result["status"] == "FAIL"  # No claim that a recovery attempt must work.
    assert result["reload_max_abs_error"] == 0.0


def test_final_evaluation_cache_refused_by_training(tmp_path):
    from tiny_omni_decision.student.data import Cache
    from tiny_omni_decision.student.recovery import run_recovery
    from tiny_omni_decision.student.ternary import TernaryController
    m = model()
    with pytest.raises(ValueError, match="role"):
        run_recovery(m, TernaryController(m),
                     Cache("evaluation", "t", [record("1")], "a"*64),
                     Cache("validation", "t", [record("2")], "b"*64),
                     tmp_path / "bad", {}, metadata={})
    assert not (tmp_path / "bad").exists()


def test_artifact_size_limit_cannot_report_pass(tmp_path):
    from tiny_omni_decision.student.data import Cache
    from tiny_omni_decision.student.recovery import run_recovery
    from tiny_omni_decision.student.ternary import TernaryController
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
        def forward(self, item):
            return self.linear(torch.ones(4))
    m = Toy()
    c = TernaryController(m, group_size=4)
    config = dict(qat_steps=1, lora_steps=1, evaluation_interval=1,
                  learning_rate=0.01, lora_learning_rate=0.01, seed=17,
                  lora_rank=1, lora_alpha=2.0, max_artifact_bytes=1,
                  limits=dict(max_accuracy_drop=1, max_nll_increase=100,
                              max_brier_increase=10, max_ece_increase=1))
    result = run_recovery(m, c, Cache("train", "t", [record("1")], "a"*64),
                          Cache("validation", "t", [record("2")], "b"*64),
                          tmp_path / "too-large", config, metadata={})
    assert result["status"] == "FAIL"
    assert not result["size_gate_passed"]
    assert not result["lora_used"]  # Adding bytes cannot fix size-only failure.


def test_evaluation_guard_checks_training_and_selection_content():
    from tiny_omni_decision.student.data import (
        Cache, assert_heldout, split_identities,
    )
    train = Cache("train", "t", [record("1")], "a"*64)
    validation = Cache("validation", "t", [record("2")], "b"*64)
    guard = split_identities(train, validation)
    leaked = Cache("evaluation", "t", [record("2")], "c"*64)
    with pytest.raises(ValueError, match="overlap"):
        assert_heldout(leaked, guard)
    assert_heldout(Cache("evaluation", "t", [record("3")], "d"*64), guard)
