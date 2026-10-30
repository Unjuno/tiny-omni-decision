"""Freeze one language-only training intervention and separate development transfer."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts.binding_language import diverse_train, heldout_conditions, unused_states
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import identity
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.prepare_binding_transfer import inputs as old_inputs

CONFIG = "configs/pretrained_reuse/binding_language_v1.json"
CODE = [
    CONFIG,
    "scripts/binding_language.py",
    "scripts/prepare_binding_language.py",
    "scripts/train_binding_language.py",
    "scripts/evaluate_binding_language.py",
    "scripts/binding_transfer.py",
    "scripts/prepare_binding_transfer.py",
    "scripts/condition_binding_corpus.py",
    "scripts/condition_binding_training.py",
    "scripts/condition_binding_retention.py",
    "scripts/build_condition_binding_corpus.py",
    "scripts/evaluate_semantic_state_bridge.py",
    "scripts/semantic_state_diagnostic.py",
    "scripts/train_text_crossencoder.py",
    "scripts/train_numeric_deterministic.py",
    "scripts/summarize_condition_binding.py",
    "src/tiny_omni_decision/dataset.py",
    "tests/test_binding_language.py",
    "docs/pretrained_reuse/R2_BINDING_LANGUAGE_PROTOCOL_20261015.md",
]


def code_pins():
    subprocess.run(["git", "ls-files", "--error-unmatch", *CODE], check=True, capture_output=True)
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def inputs(config):
    pins = old_inputs(config)
    previous = Path(config["root"]) / "binding-independent-transfer-corpus-01"
    pins[str(previous / "result.json")] = config["previous_transfer_corpus_sha256"]
    verify(pins)
    manifest = read(previous / "result.json")
    if manifest["status"] != "PASS_FROZEN_INDEPENDENT_TRANSFER":
        raise ValueError("previous transfer corpus incomplete")
    pins[str(previous / "states.json")] = manifest["artifacts"]["states.json"]
    verify(pins)
    return pins


def validate_corpus(directory, digest, config, training_only=False):
    directory = Path(directory)
    pins, code = inputs(config), code_pins()
    pins[str(directory / "result.json")] = digest
    verify(pins)
    manifest = read(directory / "result.json")
    if (
        manifest["status"] != "PASS_FROZEN_BINDING_LANGUAGE"
        or manifest["config"] != config
        or manifest["input_pins"] != inputs(config)
        or manifest["code_pins"] != code
        or manifest["state_overlap"] != 0
    ):
        raise ValueError("frozen language provenance mismatch")
    pins.update(
        {
            str(directory / p): h
            for p, h in manifest["artifacts"].items()
            if not training_only or p == "train.jsonl"
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
    old = root / "condition-binding-corpus-01"
    original = lines(old / "train.jsonl")
    train = diverse_train(original)
    annotation = lines(root / "clevr4-full-coverage-seed17/train-labels.jsonl")
    if any(r["split"] != "train" for r in annotation):
        raise ValueError("non-TRAIN annotation source")
    excluded = (
        annotation
        + sum(read(root / "semantic-state-bridge-01/states.json").values(), [])
        + sum(read(old / "state-partitions.json").values(), [])
        + read(root / "binding-independent-transfer-corpus-01/states.json")
    )
    states = unused_states(excluded, 256)
    data = heldout_conditions(states)
    training_strings = {r["question"] for r in train}
    if any(
        r["question"] in training_strings
        for n, rows in data.items()
        if n not in ("known", "no_question")
        for r in rows
    ):
        raise ValueError("heldout instruction overlap")
    for a, b in zip(original, train, strict=True):
        if any(a[k] != b[k] for k in set(a) - {"id", "question"}):
            raise ValueError("non-language TRAIN change")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "states.json", states)
    for name, rows in dict(train=train, **data).items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
    verify(pins)
    verify(code)
    write(
        args.output / "result.json",
        dict(
            status="PASS_FROZEN_BINDING_LANGUAGE",
            config=config,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            input_pins=pins,
            code_pins=code,
            excluded_unique_states=len({identity(s) for s in excluded}),
            selected_unique_states=len(states),
            state_overlap=0,
            train_questions=len(train),
            train_unique_questions=len({r["id"] for r in train}),
            train_unique_states=len({r["group"] for r in train}),
            instruction_strings=len(training_strings),
            language_family_counts=dict(Counter(r["language_family"] for r in train)),
            field_counts=dict(Counter(r["task"] for r in train)),
            rows_per_condition={n: len(v) for n, v in data.items()},
            attribute_counts={f: dict(Counter(s[f] for s in states)) for f in states[0]},
            original_training_order_sha256=sha(old / "sample-order.json"),
            model_loaded=False,
            sealed_audit_loaded=False,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output / "result.json"))))


if __name__ == "__main__":
    main()
