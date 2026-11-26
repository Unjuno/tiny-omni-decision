"""Exercise immutable input pins, unused-state reservation and actual disk output."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.condition_binding_corpus import all_states
from scripts.condition_relevance_corpus import load_development
from scripts.decision_supervision import binding_rows, canonical, digest, numeric_rows, state_key
from scripts.prepare_condition_relevance import create_corpus


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "previous"
    root.mkdir()
    states = sorted(all_states(), key=lambda s: digest([
        17, "decision-supervision-binding-v1", state_key(s)]))
    numeric = [dict(id=str(i), cost=i + 1, risk=i + 2, time=i + 3) for i in range(4)]
    train = binding_rows(states[:1], "train") + numeric_rows([numeric], "train")
    validation = binding_rows(states[1:2], "validation")
    payloads = {
        "train.jsonl": train, "validation.jsonl": validation,
        "retention_train.jsonl": binding_rows(states[:1], "train"),
        "retention_validation.jsonl": validation,
    }
    for name, rows in payloads.items():
        (root / name).write_text("".join(canonical(r) + "\n" for r in rows),
                                 encoding="utf-8", newline="\n")
    (root / "excluded-state-keys.json").write_text("[]\n", encoding="utf-8")
    manifest = dict(status="PASS_FROZEN_DECISION_SUPERVISION",
                    config=dict(seed=17, binding_counts=[1, 1, 1]),
                    artifacts={p.name: sha(p) for p in root.iterdir()},
                    selection_schema={"old:test": 8})
    (root / "result.json").write_text(canonical(manifest), encoding="utf-8")
    config = dict(id="fixture", seed=17, previous_root=str(root),
                  source_pins={p.name: sha(p) for p in root.iterdir()},
                  counts=dict(train=3, calibration=1, validation=1, transfer=1),
                  expected_unused_states=9997, expected_numeric_train_rows=8)
    return config, root


def test_actual_builder_replays_and_preserves_old_bytes_without_opening_transfer(
    inputs, tmp_path, monkeypatch
):
    config, root = inputs
    forbidden = {root / "transfer.jsonl", root / "state-partitions.json"}
    read = Path.open

    def guard(path, *args, **kwargs):
        assert path not in forbidden, "old transfer payload was opened"
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    output, replay = tmp_path / "new", tmp_path / "replay"
    result = create_corpus(config, output, {}, "fixture-source")
    assert result["status"] == "PASS_FROZEN_CONDITION_RELEVANCE"
    assert result["integrity"]["unique_states"] == 6
    assert result["training"]["new_binding_rows"] == 48
    assert result["training"]["reused_numeric_rows"] == 8
    assert result["reserved_previous_states"] == 3
    assert result["old_transfer_payload_opened"] is False
    assert (output / "old_validation.jsonl").read_bytes() == (
        root / "validation.jsonl").read_bytes()
    assert (output / "legacy_train.jsonl").read_bytes() == (
        root / "retention_train.jsonl").read_bytes()
    assert len(load_development(output, "binding_train", sha(output / "result.json"))) == 48
    create_corpus(config, replay, {}, "fixture-source")
    assert {p.name: sha(p) for p in output.iterdir()} == {p.name: sha(p) for p in replay.iterdir()}
    with pytest.raises(FileExistsError):
        create_corpus(config, output, {}, "fixture-source")


def test_corrupt_source_and_undeclared_transfer_are_rejected_before_output(inputs, tmp_path):
    config, root = inputs
    (root / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        create_corpus(config, tmp_path / "bad", {}, "fixture")
    assert not (tmp_path / "bad").exists()
    config["source_pins"]["transfer.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="declared inputs"):
        create_corpus(config, tmp_path / "forbidden", {}, "fixture")


def test_fit_loader_rejects_modified_payload_and_manifest(inputs, tmp_path):
    config, _ = inputs
    output = tmp_path / "new"
    create_corpus(config, output, {}, "fixture")
    pin = sha(output / "result.json")
    with pytest.raises(ValueError, match="manifest hash"):
        load_development(output, "binding_train", "0" * 64)
    (output / "binding_train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partition hash"):
        load_development(output, "binding_train", pin)


def test_inconsistent_old_allocation_is_rejected(inputs, tmp_path):
    config, root = inputs
    wrong = json.loads((root / "result.json").read_text())
    wrong["config"]["binding_counts"] = [2, 1, 1]
    (root / "result.json").write_text(canonical(wrong), encoding="utf-8")
    config["source_pins"]["result.json"] = sha(root / "result.json")
    config["expected_unused_states"] = 9996
    with pytest.raises(ValueError, match="previous allocation"):
        create_corpus(config, tmp_path / "bad", {}, "fixture")


def test_fit_loader_checks_split_even_for_hash_consistent_payload(inputs, tmp_path):
    config, _ = inputs
    output = tmp_path / "new"
    create_corpus(config, output, {}, "fixture")
    path = output / "binding_train.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["split"] = "transfer"
    path.write_text("".join(canonical(r) + "\n" for r in values), encoding="utf-8")
    manifest = json.loads((output / "result.json").read_text())
    manifest["artifacts"][path.name] = sha(path)
    (output / "result.json").write_text(canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="membership"):
        load_development(output, "binding_train", sha(output / "result.json"))
