import hashlib
import json
from pathlib import Path

import pytest

from scripts.condition_binding_corpus import all_states
from scripts.decision_supervision import binding_rows, state_key
from scripts.prepare_decision_supervision import create_corpus, legacy_gate


def source(tmp_path, name, data, jsonl=False):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in data) if jsonl else json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fixture(tmp_path):
    prior = all_states()[:2]
    legacy_train = binding_rows(prior[:1], "train")
    legacy_val = binding_rows(prior[1:], "validation")
    pins = {
        "excluded.json": source(tmp_path, "excluded.json", [state_key(s) for s in prior]),
        "observed.json": source(tmp_path, "observed.json", []),
        "old.jsonl": source(tmp_path, "old.jsonl", [], True),
        "replay.jsonl": source(tmp_path, "replay.jsonl", legacy_train, True),
        "legacy.jsonl": source(tmp_path, "legacy.jsonl", legacy_val, True),
    }
    return dict(root=str(tmp_path), seed=17, binding_counts=[2, 1, 1], numeric_counts=[2, 1, 1],
                binding_exclusion_ids="excluded.json", binding_observed_states="observed.json",
                numeric_sources=["old.jsonl"], retention_train="replay.jsonl",
                retention_validation=["legacy.jsonl"], source_pins=pins)


def test_real_build_hashes_replay_and_source_protection(tmp_path):
    config = config_fixture(tmp_path)
    a, b = tmp_path / "out-a", tmp_path / "out-b"
    first = create_corpus(config, a, {}, "a" * 40)
    second = create_corpus(config, b, {}, "a" * 40)
    assert first["status"] == "PASS_FROZEN_DECISION_SUPERVISION"
    assert first["artifacts"] == second["artifacts"]
    assert first["new_partitions"]["train"] == dict(rows=48, states=4)
    assert first["new_partitions"]["validation"] == dict(rows=16, states=2)
    for name, digest in first["artifacts"].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == digest
    assert (a / "retention_train.jsonl").read_bytes() == (tmp_path / "replay.jsonl").read_bytes()
    assert (a / "legacy-0.jsonl").read_bytes() == (tmp_path / "legacy.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        create_corpus(config, a, {}, "a" * 40)
    assert (a / "result.json").read_bytes() == (b / "result.json").read_bytes()


def test_changed_source_and_legacy_overlap_refuse_output(tmp_path):
    config = config_fixture(tmp_path)
    source(tmp_path, "replay.jsonl", [], True)
    with pytest.raises(ValueError, match="hash"):
        create_corpus(config, tmp_path / "failed", {}, "a" * 40)
    assert not (tmp_path / "failed").exists()


def test_legacy_train_validation_overlap_is_rejected(tmp_path):
    config = config_fixture(tmp_path)
    path = Path(config["root"]) / "replay.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for row in rows:
        row["split"] = "validation"
    config["source_pins"]["legacy.jsonl"] = source(tmp_path, "legacy.jsonl", rows, True)
    with pytest.raises(ValueError, match="legacy state overlap"):
        create_corpus(config, tmp_path / "failed", {}, "a" * 40)


def test_known_exclusion_count_cannot_silently_shrink(tmp_path):
    config = config_fixture(tmp_path)
    config["expected_exclusions"] = dict(binding=3, numeric=0)
    with pytest.raises(ValueError, match="exclusion inventory"):
        create_corpus(config, tmp_path / "failed", {}, "a" * 40)


def test_legacy_prose_is_allowed_but_normalized_overlap_rejected():
    train = [dict(id="t", split="train", state="First observation.",
                  options=["yes", "no"], target="yes")]
    validation = [dict(id="v", split="validation", state="Different observation.",
                       options=["yes", "no"], target="no")]
    legacy_gate(train, validation)
    validation[0]["state"] = "  First  observation.\n"
    with pytest.raises(ValueError, match="legacy state overlap"):
        legacy_gate(train, validation)
