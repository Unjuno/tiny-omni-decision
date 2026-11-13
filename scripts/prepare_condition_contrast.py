"""Freeze new state/syntax partitions and a matched conditional-contrast fit."""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.binding_replay_coverage import verify_split_integrity
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import identity
from scripts.condition_contrast import (
    build_rows,
    grouped_schedule,
    partition_states,
    validate_group,
)
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.prepare_binding_replay_coverage import CONFIG as PARENT_CONFIG
from scripts.prepare_binding_replay_coverage import validate_corpus as validate_parent

CONFIG = "configs/pretrained_reuse/condition_contrast_v1.json"
CODE = [
    CONFIG,
    "scripts/condition_contrast.py",
    "scripts/prepare_condition_contrast.py",
    "scripts/train_condition_contrast.py",
    "scripts/evaluate_condition_contrast.py",
    "tests/test_condition_contrast.py",
    "docs/pretrained_reuse/R2_CONDITION_CONTRAST_PROTOCOL_20261015.md",
]


def code_pins():
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *CODE], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def inputs(config):
    root = Path(config["root"])
    parent = read(PARENT_CONFIG)
    pins, code = validate_parent(
        root / "binding-replay-coverage-corpus-01", config["replay_corpus_sha256"], parent, True
    )
    pins.update(code)
    fit = root / "binding-replay-coverage-fit-01"
    pins[str(fit / "result.json")] = config["replay_fit_sha256"]
    pins[str(root / "binding-language-corpus-01/states.json")] = config["language_states_sha256"]
    verify(pins)
    manifest = read(fit / "result.json")
    if manifest["status"] != "PASS_FIXED_REPLAY_COVERAGE":
        raise ValueError("completed parent cache source required")
    pins[str(fit / "teacher-cache.json")] = manifest["artifacts"]["teacher-cache.json"]
    verify(pins)
    return pins


def validate_corpus(directory, digest, config, training_only=False):
    directory = Path(directory)
    pins, code = inputs(config), code_pins()
    pins[str(directory / "result.json")] = digest
    verify(pins)
    m = read(directory / "result.json")
    if (
        m["status"] != "PASS_FROZEN_CONDITION_CONTRAST"
        or m["config"] != config
        or m["code_pins"] != code
        or m["input_pins"] != inputs(config)
        or m["state_overlap"] != 0
    ):
        raise ValueError("frozen contrast provenance mismatch")
    permitted = {"train.jsonl", "validation.jsonl", "replay.jsonl", "sample-order.json"}
    pins.update(
        {
            str(directory / p): h
            for p, h in m["artifacts"].items()
            if not training_only or p in permitted
        }
    )
    verify(pins)
    return pins, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    pins, code = inputs(config), code_pins()
    root = Path(config["root"])
    excluded = (
        lines(root / "clevr4-full-coverage-seed17/train-labels.jsonl")
        + sum(read(root / "semantic-state-bridge-01/states.json").values(), [])
        + sum(read(root / "condition-binding-corpus-01/state-partitions.json").values(), [])
        + read(root / "binding-independent-transfer-corpus-01/states.json")
        + read(root / "binding-language-corpus-01/states.json")
    )
    parts = partition_states(excluded, config["partition_counts"])
    train = build_rows(parts["train"], "train")
    validation = build_rows(parts["validation"], "validation")
    transfer = build_rows(parts["transfer"], "transfer")
    replay = lines(root / "binding-replay-coverage-corpus-01/replay.jsonl")
    data = dict(train=train, validation=validation, **transfer)
    for rows in data.values():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate row")
        for i in range(0, len(rows), 4):
            validate_group(rows[i : i + 4])
    trq = {r["question"] for r in train}
    vaq = {r["question"] for r in validation}
    teq = {
        r["question"]
        for n, rows in transfer.items()
        if n not in ("known", "no_question")
        for r in rows
    }
    if trq & vaq or trq & teq or vaq & teq:
        raise ValueError("syntax literal overlap")
    legacy = sum(
        (
            lines(root / "condition-binding-corpus-01" / f"{n}.jsonl")
            for n in ("validation", "original", "curriculum")
        ),
        [],
    )
    verify_split_integrity(train + replay, validation + transfer["known"] + legacy)
    for a, b in (("train", "validation"), ("train", "transfer"), ("validation", "transfer")):
        if {identity(s) for s in parts[a]} & {identity(s) for s in parts[b]}:
            raise ValueError("state overlap")
    old_order = read(root / "binding-replay-coverage-corpus-01/sample-order.json")
    order = grouped_schedule(old_order)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in data.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
    shutil.copyfile(
        root / "binding-replay-coverage-corpus-01/replay.jsonl", args.output / "replay.jsonl"
    )
    write(args.output / "state-partitions.json", parts)
    write(args.output / "sample-order.json", order)
    write(
        args.output / "accounting.json",
        [
            dict(
                step=step,
                consumed=8 * step,
                unique_questions=len({r["id"] for r in used}),
                source_counts=dict(Counter(r["source"] for r in used)),
                unique_states_by_source={
                    s: len({r["group"] for r in used if r["source"] == s})
                    for s in ("binding", "original", "curriculum")
                },
            )
            for step in (0, 128, 256, 384, 512)
            for used in [[([train, replay][s])[i] for batch in order[:step] for s, i in batch]]
        ],
    )
    verify(pins)
    verify(code)
    write(
        args.output / "result.json",
        dict(
            status="PASS_FROZEN_CONDITION_CONTRAST",
            config=config,
            input_pins=pins,
            code_pins=code,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            excluded_unique_states=len({identity(s) for s in excluded}),
            state_overlap=0,
            partitions={k: len(v) for k, v in parts.items()},
            rows={k: len(v) for k, v in data.items()},
            train_families=dict(Counter(r["language_family"] for r in train)),
            native_media_assets=0,
            sealed_audit_loaded=False,
            model_loaded=False,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output / "result.json"))))


if __name__ == "__main__":
    main()
