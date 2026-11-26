"""Freeze independent decision supervision without loading any model or audit."""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.condition_binding_corpus import all_states
from scripts.decision_supervision import (
    SPLITS,
    binding_rows,
    canonical,
    digest,
    numeric_rows,
    state_key,
    verify_partitions,
)

CONFIG = "configs/pretrained_reuse/decision_supervision_v1.json"
CODE = [
    CONFIG,
    "scripts/decision_supervision.py",
    "scripts/prepare_decision_supervision.py",
    "scripts/condition_binding_corpus.py",
    "scripts/semantic_state_diagnostic.py",
    "src/tiny_omni_decision/dataset.py",
    "tests/test_decision_supervision.py",
    "tests/test_prepare_decision_supervision.py",
    "docs/pretrained_reuse/R2_DECISION_SUPERVISION_PROTOCOL_20261015.md",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")


def split_values(values, counts):
    if len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("three positive state counts required")
    if len(values) < sum(counts):
        raise ValueError("insufficient unused states")
    result, start = {}, 0
    for name, count in zip(SPLITS, counts, strict=True):
        result[name] = values[start:start + count]
        start += count
    return result


def metric_key(row, legacy=False):
    if not legacy:
        return row["task"]
    task = row["task"]
    if row.get("source") == "curriculum" and task in ("extremum", "temporal_order"):
        task = f"{row['oracle']['field']}:{row['oracle']['direction']}"
    return f"legacy:{row.get('source', 'fixture')}:{task}"


def legacy_gate(train, validation):
    for rows, split in ((train, "train"), (validation, "validation")):
        if not rows or any(r["split"] != split for r in rows):
            raise ValueError("legacy partition mismatch")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate legacy IDs")
        if any(r["target"] not in r["options"] for r in rows):
            raise ValueError("legacy target absent")
    if {state_key(r["state"]) for r in train} & {state_key(r["state"]) for r in validation}:
        raise ValueError("legacy state overlap")
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("legacy ID overlap")


def create_corpus(config, output, code_pins, source_commit):
    output, root = Path(output), Path(config["root"])
    if output.exists():
        raise FileExistsError(output)
    pins = {str(root / name): value for name, value in config["source_pins"].items()}
    verify(pins)
    verify(code_pins)
    if config["seed"] != 17:
        raise ValueError("immutable primary seed17 required")
    declared = [config["binding_exclusion_ids"], config["binding_observed_states"],
                config["retention_train"], *config["numeric_sources"],
                *config["retention_validation"]]
    if set(declared) != set(config["source_pins"]):
        raise ValueError("all and only declared inputs must be hash-pinned")
    binding_excluded = set(read(root / config["binding_exclusion_ids"]))
    binding_excluded.update(state_key(s) for s in read(root / config["binding_observed_states"]))
    numeric_excluded = set()
    source_counts = {}
    for name in config["numeric_sources"]:
        rows = lines(root / name)
        keys = {state_key(r["state"]) for r in rows}
        numeric_excluded.update(keys)
        source_counts[name] = dict(rows=len(rows), state_classes=len(keys))
    if "expected_exclusions" in config and config["expected_exclusions"] != dict(
        binding=len(binding_excluded), numeric=len(numeric_excluded)
    ):
        raise ValueError("historical exclusion inventory mismatch")
    legacy_train = lines(root / config["retention_train"])
    legacy_validation = [r for name in config["retention_validation"] for r in lines(root / name)]
    legacy_gate(legacy_train, legacy_validation)
    excluded = binding_excluded | numeric_excluded | {
        state_key(r["state"]) for r in legacy_train + legacy_validation}
    pool = sorted((s for s in all_states() if state_key(s) not in excluded),
                  key=lambda s: digest([17, "decision-supervision-binding-v1", state_key(s)]))
    binding_parts = split_values(pool, config["binding_counts"])
    rng = random.Random("17:decision-supervision-numeric-v1")
    numeric, used, rejected = [], set(excluded), 0
    needed = sum(config["numeric_counts"])
    for _ in range(needed * 10):
        labels = rng.sample(list("KMPRTVXZ"), 4)
        columns = {key: rng.sample(range(1, 200), 4) for key in ("cost", "risk", "time")}
        state = [dict(id=label, **{k: v[i] for k, v in columns.items()})
                 for i, label in enumerate(labels)]
        key = state_key(state)
        if key in used:
            rejected += 1
            continue
        numeric.append(state)
        used.add(key)
        if len(numeric) == needed:
            break
    numeric_parts = split_values(numeric, config["numeric_counts"])
    parts = {name: binding_rows(binding_parts[name], name) + numeric_rows(numeric_parts[name], name)
             for name in SPLITS}
    integrity = verify_partitions(parts, excluded)
    selection_schema = dict(sorted(Counter(
        [metric_key(r) for r in parts["validation"]]
        + [metric_key(r, True) for r in legacy_validation]).items()))
    accounting = {}
    for name, rows in parts.items():
        accounting[name] = dict(
            rows=len(rows), unique_ids=len({r["id"] for r in rows}),
            unique_states=len({r["group"] for r in rows}),
            tasks=dict(sorted(Counter(r["task"] for r in rows).items())),
            candidate_counts=dict(sorted(Counter(len(r["options"]) for r in rows).items())),
            target_positions=dict(sorted(Counter(r["option_ids"].index(r["target_id"])
                                                  for r in rows).items())),
            templates=len({r["question"] for r in rows}),
            numeric_count_targets=dict(sorted(Counter(r["target_id"] for r in rows
                if r["task"] == "numeric:count").items())),
            numeric_none_targets=sum(r["target_id"] == "none" for r in rows),
        )
    verify(pins)
    verify(code_pins)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in parts.items():
        (output / f"{name}.jsonl").write_text(
            "".join(canonical(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    shutil.copyfile(root / config["retention_train"], output / "retention_train.jsonl")
    combined = b""
    for i, name in enumerate(config["retention_validation"]):
        raw = (root / name).read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("legacy JSONL must end with newline")
        shutil.copyfile(root / name, output / f"legacy-{i}.jsonl")
        combined += raw
    (output / "retention_validation.jsonl").write_bytes(combined)
    values = dict(
        **{"accounting.json": accounting, "selection-schema.json": selection_schema},
        **{"state-partitions.json": dict(binding=binding_parts, numeric=numeric_parts),
           "excluded-state-keys.json": sorted(excluded)},
    )
    for name, value in values.items():
        (output / name).write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")
    result = dict(
        status="PASS_FROZEN_DECISION_SUPERVISION", source_commit=source_commit, config=config,
        input_pins=pins, code_pins=code_pins, new_partitions=integrity["partitions"],
        integrity=integrity, excluded_binding_states=len(binding_excluded),
        excluded_numeric_state_classes=len(numeric_excluded),
        total_excluded_state_classes=len(excluded), numeric_collision_rejections=rejected,
        historical_numeric_sources=source_counts,
        selection_schema=selection_schema,
        retention=dict(train_rows=len(legacy_train), validation_rows=len(legacy_validation),
                       train_states=len({state_key(r["state"]) for r in legacy_train}),
                       validation_states=len({state_key(r["state"]) for r in legacy_validation})),
        native_assets=0, model_loaded=False, optimizer_updates=0, sealed_audit_loaded=False,
        model_quality_evaluated=False, product_complete=False,
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    verify(pins)
    verify(code_pins)
    (output / "result.json").write_text(canonical(result) + "\n", encoding="utf-8", newline="\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *CODE],
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *CODE],
                   check=True, capture_output=True)
    code = {str(Path(p).resolve()): sha(p) for p in CODE}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = create_corpus(read(CONFIG), args.output, code, commit)
    print(json.dumps(dict(output=str(args.output), status=result["status"],
                          sha256=sha(args.output / "result.json"))))


if __name__ == "__main__":
    main()
