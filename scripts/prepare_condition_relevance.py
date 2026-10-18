"""Build a separate condition relevance corpus without opening old transfer."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.condition_binding_corpus import all_states
from scripts.condition_relevance_corpus import SPLITS, make_rows, unused_states, validate_parts
from scripts.decision_supervision import canonical, digest, state_key

CONFIG = "configs/pretrained_reuse/condition_relevance_v1.json"
SOURCE_NAMES = {
    "result.json", "excluded-state-keys.json", "train.jsonl", "validation.jsonl",
    "retention_train.jsonl", "retention_validation.jsonl",
}
CODE = [CONFIG, "scripts/condition_relevance_corpus.py", "scripts/prepare_condition_relevance.py",
        "scripts/condition_binding_corpus.py", "scripts/decision_supervision.py",
        "scripts/semantic_state_diagnostic.py", "src/tiny_omni_decision/dataset.py",
        "tests/test_condition_relevance_corpus.py", "tests/test_prepare_condition_relevance.py",
        "docs/pretrained_reuse/R2_CONDITION_RELEVANCE_DATA_PROTOCOL_20261015.md"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(pins):
    for name, expected in pins.items():
        if sha(name) != expected:
            raise ValueError(f"source hash mismatch: {name}")


def rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def write_json(path, value):
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def global_groups(parts, numeric, legacy_train, old_validation, legacy_validation):
    groups = {split: {r["group"] for r in values} for split, values in parts.items()}
    groups["train"].update(state_key(r["state"]) for r in numeric + legacy_train)
    groups["validation"].update(state_key(r["state"]) for r in old_validation + legacy_validation)
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if groups[first] & groups[second]:
                raise ValueError(f"reused data cross-partition state overlap: {first}/{second}")
    return {split: len(values) for split, values in groups.items()}


def create_corpus(config, output, code_pins, source_commit):
    root, output = Path(config["previous_root"]), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if set(config["source_pins"]) != SOURCE_NAMES:
        raise ValueError("all and only declared inputs are permitted; no old transfer access")
    if config["seed"] != 17:
        raise ValueError("immutable primary seed17 required")
    counts = config["counts"]
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("four positive state counts required")
    pins = {str(root / name): pin for name, pin in config["source_pins"].items()}
    verify(pins)
    verify(code_pins)
    previous = read(root / "result.json")
    if (previous["status"] != "PASS_FROZEN_DECISION_SUPERVISION"
            or previous["config"]["seed"] != 17):
        raise ValueError("verified previous corpus and seed required")
    if any(previous["artifacts"].get(name) != config["source_pins"][name]
           for name in SOURCE_NAMES - {"result.json"}):
        raise ValueError("previous manifest payload hash mismatch")
    excluded = set(read(root / "excluded-state-keys.json"))
    old_counts = previous["config"]["binding_counts"]
    universe = all_states()
    available, reserved = unused_states(universe, excluded, old_counts)
    if len(available) != config["expected_unused_states"]:
        raise ValueError("unused state inventory mismatch")
    old_train, old_validation = rows(root / "train.jsonl"), rows(root / "validation.jsonl")
    old_order = sorted((s for s in universe if state_key(s) not in excluded),
                       key=lambda s: digest([17, "decision-supervision-binding-v1", state_key(s)]))
    start = 0
    for values, count in zip((old_train, old_validation), old_counts[:2], strict=True):
        actual = {state_key(r["state"]) for r in values if r["task"].startswith("binding:")}
        expected = {state_key(s) for s in old_order[start:start + count]}
        if actual != expected:
            raise ValueError("previous allocation does not match the pinned reservation recipe")
        start += count
    if sum(counts.values()) > len(available):
        raise ValueError("insufficient previously unused states")
    available.sort(key=lambda s: digest([17, "condition-relevance-state-v1", state_key(s)]))
    parts, identities, start = {}, {}, 0
    for split in SPLITS:
        selected = available[start:start + counts[split]]
        parts[split] = [r for index, state in enumerate(selected)
                        for r in make_rows(state, split, index)]
        identities[split] = [state_key(s) for s in selected]
        start += counts[split]
    integrity = validate_parts(parts, excluded | reserved)
    numeric_bytes = [line for line in (root / "train.jsonl").read_bytes().splitlines(keepends=True)
                     if json.loads(line)["task"].startswith("numeric:")]
    numeric = [json.loads(line) for line in numeric_bytes]
    if len(numeric) != config["expected_numeric_train_rows"]:
        raise ValueError("reused numeric count mismatch")
    legacy_train, legacy_validation = rows(root / "retention_train.jsonl"), rows(
        root / "retention_validation.jsonl")
    total_groups = global_groups(parts, numeric, legacy_train, old_validation, legacy_validation)
    selection_schema = dict(previous["selection_schema"])
    new_schema = Counter(f"relevance:{r['field']}:q{r['query_index']}:{r['option_form']}"
                         for r in parts["validation"])
    if set(new_schema) & set(selection_schema):
        raise ValueError("selection schema collision")
    selection_schema.update(dict(sorted(new_schema.items())))
    verify(pins)
    verify(code_pins)
    output.mkdir(parents=True, exist_ok=False)
    for split, values in parts.items():
        name = "binding_train" if split == "train" else split
        (output / f"{name}.jsonl").write_text("".join(canonical(r) + "\n" for r in values),
                                            encoding="utf-8", newline="\n")
    (output / "numeric_train.jsonl").write_bytes(b"".join(numeric_bytes))
    for old_name, new_name in (("validation.jsonl", "old_validation.jsonl"),
                               ("retention_train.jsonl", "legacy_train.jsonl"),
                               ("retention_validation.jsonl", "legacy_validation.jsonl")):
        shutil.copyfile(root / old_name, output / new_name)
    write_json(output / "state-identities.json", identities)
    write_json(output / "selection-schema.json", selection_schema)
    write_json(output / "accounting.json", integrity)
    verify(pins)
    verify(code_pins)
    result = dict(
        status="PASS_FROZEN_CONDITION_RELEVANCE", config=config,
        source_commit=source_commit, code_pins=code_pins, input_pins=pins,
        integrity=integrity, reserved_previous_states=len(reserved),
        historical_excluded_states=len(excluded), available_before_allocation=len(available),
        remaining_unused_states=len(available) - sum(counts.values()),
        all_included_state_counts=total_groups,
        training=dict(new_binding_rows=len(parts["train"]), reused_numeric_rows=len(numeric),
                      reused_legacy_rows=len(legacy_train),
                      total_rows=len(parts["train"]) + len(numeric) + len(legacy_train)),
        selection_schema=selection_schema, old_validation_unchanged=True,
        old_transfer_payload_opened=False, old_state_partitions_opened=False,
        new_transfer_constructed=True, new_transfer_model_evaluated=False,
        calibration_scope="binding_only; not sufficient for all-task calibration",
        calibration_loaded_for_fit=False, optimizer_updates=0, native_assets=0,
        model_loaded=False, model_quality_evaluated=False, sealed_audit_loaded=False,
        product_complete=False,
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0"))
    if set(CODE) - tracked:
        raise ValueError("all execution sources must be tracked")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *CODE],
                   check=True, capture_output=True)
    code_pins = {str(Path(p).resolve()): sha(p) for p in CODE}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = create_corpus(read(CONFIG), args.output, code_pins, commit)
    print(json.dumps(dict(output=str(args.output), status=result["status"],
                          result_sha256=sha(args.output / "result.json"),
                          training=result["training"])))


if __name__ == "__main__":
    main()
