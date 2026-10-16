import copy

import pytest


def video_receipt():
    return dict(
        status="PASS_COMPLETE_VIDEO_CACHE",
        phase="full",
        expected_events=11151,
        completed_events=11151,
        expected_questions=32582,
        completed_questions=32582,
        expected_parents=409,
        completed_parents=409,
        frozen_before="abc",
        frozen_after="abc",
        optimizer_updates=0,
        matched_reference_events=21,
    )


@pytest.mark.parametrize("change", ["running", "partial", "frozen", "references", "phase"])
def test_video_admission_rejects_incomplete_receipt(change):
    from scripts.greatest_hits_matched_inputs import require_video

    receipt = video_receipt()
    if change == "running":
        receipt["status"] = "RUNNING"
    if change == "partial":
        receipt["completed_events"] -= 1
    if change == "frozen":
        receipt["frozen_after"] = "different"
    if change == "references":
        receipt["matched_reference_events"] = 20
    if change == "phase":
        receipt["phase"] = "smoke"
    with pytest.raises(ValueError):
        require_video(receipt)


def test_video_admission_accepts_complete_receipt():
    from scripts.greatest_hits_matched_inputs import require_video

    require_video(video_receipt())


def test_incomplete_video_ledger_rejected_before_any_input_hashing(monkeypatch):
    from scripts import greatest_hits_matched_inputs as inputs

    value = dict(
        video_receipt(),
        parents=[{"parent": "parent"}],
        artifacts={"parent-positions.npy": "a" * 64, "parent-pts.npy": "b" * 64},
    )
    monkeypatch.setattr(inputs, "read", lambda path: value)

    class NoPin:
        def add(self, *args):
            raise AssertionError("incomplete video must be rejected before input hashing")

    with pytest.raises(ValueError, match="ledger"):
        inputs.audit(NoPin())


@pytest.mark.parametrize("change", ["missing_video", "missing_positions", "missing_pts", "extra"])
def test_exact_complete_video_inventory(change):
    from scripts.greatest_hits_matched_inputs import require_video_inventory

    value = dict(
        parents=[{"parent": "parent"}],
        artifacts={"parent-" + key + ".npy": "a" * 64 for key in ("video", "positions", "pts")},
    )
    require_video_inventory(value)
    if change == "extra":
        value["artifacts"]["unknown.npy"] = "a" * 64
    else:
        value["artifacts"].pop("parent-" + change.removeprefix("missing_") + ".npy")
    with pytest.raises(ValueError, match="ledger"):
        require_video_inventory(value)


def test_pin_verifies_original_bytes_and_catches_cached_changes(tmp_path):
    from scripts.greatest_hits_matched_inputs import Pins
    from scripts.native_video_cache import sha

    path = tmp_path / "bank.npy"
    path.write_bytes(b"original")
    pins = Pins()
    pins.add(path, sha(path))
    pins.check()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        pins.check()
    with pytest.raises(ValueError):
        pins.verify()


def metrics(nll=1.0, accuracy=0.7):
    return dict(
        macro=dict(nll=nll),
        modality_macro={m: dict(accuracy=accuracy) for m in ("text", "image", "audio", "video")},
    )


def test_selection_late_failure_keeps_best_and_final_separate(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_greatest_hits_matched import Selection

    selected = Selection()
    common = metrics()
    path = tmp_path / "checkpoint-0000.pt"
    path.write_bytes(b"initial")
    assert selected.consider(0, 2.0, common)["selected"]
    selected.save(path, tmp_path)
    path = tmp_path / "checkpoint-0512.pt"
    path.write_bytes(b"best")
    assert selected.consider(512, 1.0, common)["selected"]
    selected.save(path, tmp_path)
    path = tmp_path / "checkpoint-32768.pt"
    path.write_bytes(b"final-ineligible")
    assert not selected.consider(32768, 0.1, metrics(nll=2.0))["selected"]
    selected.save(path, tmp_path, final=True)
    assert (tmp_path / "best.pt").read_bytes() == b"best"
    assert (tmp_path / "final.pt").read_bytes() == b"final-ineligible"
    assert selected.best_step == 512


def test_model_only_warm_start_and_exact_fixed_config():
    pytest.importorskip("torch")
    from scripts.train_greatest_hits_matched import configuration

    config = configuration("E1G1")
    assert config["max_steps"] == 32768
    assert config["microbatches"] == 8
    assert config["seed"] == 17
    assert config["initialization"] == "fitted control8192 model; fresh optimizer"
    assert config["performance_early_stopping"] is False
    assert config["evaluation_steps"] == [
        0,
        512,
        1024,
        2048,
        4096,
        8192,
        12288,
        16384,
        24576,
        32768,
    ]
    with pytest.raises(ValueError):
        configuration("unknown")


@pytest.mark.parametrize("changed", ["arm", "source_commit", "input_digest", "config", "status"])
def test_train_reference_rejects_wrong_smoke(changed):
    pytest.importorskip("torch")
    from scripts.train_greatest_hits_matched import require_reference

    current = dict(arm="C0", source_commit="source", input_digest="input", config={"seed": 17})
    reference = dict(current, status="PASS_MATCHED_AV_SMOKE", projected_full_seconds=100)
    broken = copy.deepcopy(reference)
    broken[changed] = "wrong"
    with pytest.raises(ValueError):
        require_reference("train", current, broken, 200)
    require_reference("train", current, reference, 200)
    with pytest.raises(ValueError):
        require_reference("train", current, reference, 50)


def test_cpu_cli_smoke_train_and_checkpoint_replay(tmp_path, monkeypatch):
    """Synthetic CPU plumbing test; not full corpus/GPU admission evidence."""
    import json
    import sys
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from scripts import train_greatest_hits_matched as runner
    from scripts.greatest_hits_matched import digest
    from scripts.native_video_cache import sha
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    train = [
        dict(
            id=f"j{i}",
            event_id=f"e{i}",
            options=["a", "b"],
            target=i % 2,
            task="material",
            split="train",
        )
        for i in range(4)
    ]
    common = [
        dict(
            id=f"c{i}",
            options=["a", "b"],
            target_index=i % 2,
            queries=["a", "b"],
            source=m,
            task="choice",
            modality=m,
            split="train",
            observation=f"o{i}",
        )
        for i, m in enumerate(("text", "image", "audio", "video"))
    ]
    data = {("joint", "train"): train, ("common", "train"): common}
    for kind in ("joint", "common"):
        data[kind, "validation"] = [
            dict(r, id="v" + r["id"], split="validation") for r in data[kind, "train"]
        ]
    preflight, audit_dir = tmp_path / "preflight", tmp_path / "audit"
    preflight.mkdir()
    audit_dir.mkdir()
    plan = {"order": [[r["id"] for r in train + common]] * 2}
    plan["order_sha256"] = digest(plan["order"])
    runner.write(preflight / "sampling.json", plan)
    runner.write(preflight / "coverage.json", {str(i): {"consumed": i * 8} for i in range(3)})
    torch.save(runner.create_model("C0").state_dict(), preflight / "C0-initial.pt")
    proof = dict(input_pins={}, input_digest=digest({}), binding={})
    runner.write(audit_dir / "admission.json", proof)
    runner.write(
        audit_dir / "result.json",
        dict(
            status="PASS_MATCHED_AV_INPUT_AUDIT",
            input_digest=proof["input_digest"],
            artifacts={"admission.json": sha(audit_dir / "admission.json")},
        ),
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixed test protocol")
    for name, value in [
        ("PREFLIGHT", preflight),
        ("PROTOCOL", protocol),
        ("MAX_STEPS", 2),
        ("SMOKE_STEPS", 2),
        ("EVALUATION_STEPS", (0, 1, 2)),
        ("PROBE_COUNTS", {"joint": 4, "common": 4}),
    ]:
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner, "metadata", lambda: data)
    original_read = runner.read
    monkeypatch.setattr(
        runner,
        "read",
        lambda p: (
            original_read(p)
            if p.parent in (preflight, audit_dir) or str(p).startswith(str(tmp_path / "run-"))
            else {"artifacts": {}}
        ),
    )
    monkeypatch.setattr(runner, "common_banks", lambda: {})
    blocks = [
        NativeTokenBlock(
            k, torch.ones(3, width), torch.zeros(3, 3), torch.ones(3, dtype=torch.bool)
        )
        for k, width in [("speech", 384), ("video", 768)]
    ]
    q = torch.stack([torch.zeros(384), torch.ones(384)])

    class Joint:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, ident, *, order=None, mode="joint"):
            available = blocks if mode == "joint" else blocks[:1] if mode == "audio" else blocks[1:]
            return available, q[list(range(2)) if order is None else order]

        def verify_files(self):
            pass

    class Common(Joint):
        def get(self, row):
            return blocks[:1], q[[0 if k == "a" else 1 for k in row["queries"]]]

    monkeypatch.setattr(runner, "ExplicitEventStore", Joint)
    monkeypatch.setattr(runner, "FeatureStore", Common)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    for name in ("set_num_threads", "set_num_interop_threads"):
        monkeypatch.setattr(torch, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test stub")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        runner.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(peak_wset=0)),
    )
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 2**30))

    def command(args, **kwargs):
        value = (
            "0,0" if args[0] == "nvidia-smi" else "test-source" if args[1] == "rev-parse" else ""
        )
        return value if kwargs.get("text") else value.encode()

    monkeypatch.setattr(runner.subprocess, "check_output", command)
    previous = None
    for phase in ("smoke", "train", "replay"):
        output = tmp_path / ("run-" + phase)
        argv = [
            "runner",
            "--phase",
            phase,
            "--arm",
            "C0",
            "--output",
            str(output),
            "--audit",
            str(audit_dir),
            "--audit-sha256",
            sha(audit_dir / "result.json"),
            "--max-seconds",
            "3600",
        ]
        if previous:
            argv += [
                "--reference",
                str(previous),
                "--reference-sha256",
                sha(previous / "result.json"),
            ]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "PASS_MATCHED_AV_" + phase.upper()
        if phase == "train":
            assert result["best_final_reload_verified"]
            assert result["full_accounting"]["consumed"] == 16
            assert len((output / "training-history.jsonl").read_text().splitlines()) == 2
            assert (output / "final.pt").read_bytes() == (
                output / "checkpoint-00002.pt"
            ).read_bytes()
        previous = output
