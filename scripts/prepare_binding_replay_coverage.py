"""Freeze a 1024-original / 1024-curriculum TRAIN replay comparison."""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.binding_replay_coverage import (
    mixed_replay,
    verify_intervention,
    verify_split_integrity,
)
from scripts.build_condition_binding_corpus import lines
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.prepare_binding_language import CONFIG as LANGUAGE_CONFIG
from scripts.prepare_binding_language import validate_corpus as validate_language

CONFIG = "configs/pretrained_reuse/binding_replay_coverage_v1.json"
CODE = [
    CONFIG,
    "scripts/binding_replay_coverage.py",
    "scripts/prepare_binding_replay_coverage.py",
    "scripts/train_binding_replay_coverage.py",
    "scripts/evaluate_binding_replay_coverage.py",
    "tests/test_binding_replay_coverage.py",
    "docs/pretrained_reuse/R2_BINDING_REPLAY_COVERAGE_PROTOCOL_20261015.md",
]


def code_pins():
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *CODE], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def inputs(config, training_only=True):
    parent = read(LANGUAGE_CONFIG)
    verify_intervention(config, parent)
    root = Path(config["root"])
    pins, old_code = validate_language(
        root / "binding-language-corpus-01", config["language_corpus_sha256"], parent, training_only
    )
    pins.update(old_code)
    control = root / "binding-language-fit-01"
    pins[str(control / "result.json")] = config["language_fit_sha256"]
    pins[str(root / "conditional-curriculum-01/train.jsonl")] = config["curriculum_train_sha256"]
    verify(pins)
    fit = read(control / "result.json")
    if fit["status"] != "PASS_FIXED_BINDING_LANGUAGE" or fit["steps"] != 512:
        raise ValueError("complete original language comparison required")
    pins.update({str(control / p): h for p, h in fit["artifacts"].items()})
    verify(pins)
    return pins


def validate_corpus(directory, digest, config, training_only=False):
    directory = Path(directory)
    pins, code = inputs(config, training_only), code_pins()
    pins[str(directory / "result.json")] = digest
    verify(pins)
    manifest = read(directory / "result.json")
    if (
        manifest["status"] != "PASS_FROZEN_REPLAY_COVERAGE"
        or manifest["config"] != config
        or manifest["code_pins"] != code
        or manifest["input_pins"] != inputs(config, True)
        or not manifest["disjoint"]
    ):
        raise ValueError("frozen replay provenance mismatch")
    pins.update({str(directory / p): h for p, h in manifest["artifacts"].items()})
    verify(pins)
    return pins, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    pins, code = inputs(config), code_pins()
    root = Path(config["root"])
    old, language = root / "condition-binding-corpus-01", root / "binding-language-corpus-01"
    order = read(old / "sample-order.json")
    replay = mixed_replay(
        lines(old / "replay.jsonl"), lines(root / "conditional-curriculum-01/train.jsonl"), order
    )
    train = lines(language / "train.jsonl")
    validation = sum(
        (lines(old / f"{n}.jsonl") for n in ("validation", "original", "curriculum")), []
    )
    verify_split_integrity(train + replay, validation)
    # Existing binding transfer states are structurally different from curriculum/original;
    # the old corpus already gates all binding TRAIN/transfer states. No heldout payload opened.
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(language / "train.jsonl", args.output / "train.jsonl")
    shutil.copyfile(old / "sample-order.json", args.output / "sample-order.json")
    (args.output / "replay.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in replay), encoding="utf-8"
    )
    accounting = []
    for step in (0, 128, 256, 384, 512):
        used = [([train, replay][s])[i] for batch in order[:step] for s, i in batch]
        accounting.append(
            dict(
                step=step,
                consumed=len(used),
                unique_questions=len({r["id"] for r in used}),
                source_counts=dict(Counter(r["source"] for r in used)),
                unique_states_by_source={
                    s: len({r["group"] for r in used if r["source"] == s})
                    for s in ("binding", "original", "curriculum")
                },
                task_counts=dict(Counter(r["source"] + ":" + r["task"] for r in used)),
            )
        )
    write(args.output / "accounting.json", accounting)
    verify(pins)
    verify(code)
    write(
        args.output / "result.json",
        dict(
            status="PASS_FROZEN_REPLAY_COVERAGE",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config=config,
            input_pins=pins,
            code_pins=code,
            disjoint=True,
            language_train_unchanged=sha(args.output / "train.jsonl")
            == sha(language / "train.jsonl"),
            sample_order_unchanged=sha(args.output / "sample-order.json")
            == sha(old / "sample-order.json"),
            model_loaded=False,
            sealed_audit_loaded=False,
            transfer_payload_loaded=False,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output / "result.json"))))


if __name__ == "__main__":
    main()
