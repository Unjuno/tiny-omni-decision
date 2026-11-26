"""Freeze unused state combinations and all instruction conditions before inference."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts.binding_transfer import build_conditions, select_states
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import identity
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write

CONFIG = "configs/pretrained_reuse/binding_transfer_v1.json"
CODE = [
    CONFIG,
    "scripts/binding_transfer.py",
    "scripts/prepare_binding_transfer.py",
    "scripts/evaluate_binding_transfer.py",
    "scripts/condition_binding_corpus.py",
    "scripts/condition_binding_training.py",
    "scripts/build_condition_binding_corpus.py",
    "scripts/semantic_state_diagnostic.py",
    "scripts/evaluate_semantic_state_bridge.py",
    "scripts/summarize_condition_binding.py",
    "scripts/train_numeric_deterministic.py",
    "src/tiny_omni_decision/dataset.py",
    "tests/test_binding_transfer.py",
    "docs/pretrained_reuse/R2_BINDING_TRANSFER_PROTOCOL_20261015.md",
]


def inputs(config):
    root = Path(config["root"])
    fit_path = root / "condition-binding-retention-fit-01/result.json"
    corpus_path = root / "condition-binding-corpus-01/result.json"
    pins = {
        str(fit_path): config["fit_result_sha256"],
        str(corpus_path): config["old_corpus_sha256"],
    }
    verify(pins)
    fit, corpus = read(fit_path), read(corpus_path)
    if fit["status"] != "PASS_FIXED_BINDING_RETENTION" or fit["best_step"] != 512:
        raise ValueError("selected fixed candidate unavailable")
    pins.update(fit["input_pins"])
    pins.update(fit["code_pins"])
    pins.update({str(fit_path.parent / p): h for p, h in fit["artifacts"].items()})
    pins.update({str(corpus_path.parent / p): h for p, h in corpus["hashes"].items()})
    verify(pins)
    return pins


def code_pins():
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *CODE], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    pins, code = inputs(config), code_pins()
    root = Path(config["root"])
    annotation = lines(root / "clevr4-full-coverage-seed17/train-labels.jsonl")
    if any(r["split"] != "train" for r in annotation):
        raise ValueError("non-TRAIN source annotations")
    previous = read(root / "semantic-state-bridge-01/states.json")
    partitions = read(root / "condition-binding-corpus-01/state-partitions.json")
    excluded = annotation + sum(previous.values(), []) + sum(partitions.values(), [])
    states = select_states(excluded, config["states"])
    data = build_conditions(states)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "states.json", states)
    for name, rows in data.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
    verify(pins)
    verify(code)
    write(
        args.output / "result.json",
        dict(
            status="PASS_FROZEN_INDEPENDENT_TRANSFER",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config=config,
            input_pins=pins,
            code_pins=code,
            excluded_unique_states=len({identity(s) for s in excluded}),
            selected_unique_states=len(states),
            state_overlap=0,
            rows_per_condition={n: len(v) for n, v in data.items()},
            attribute_counts={f: dict(Counter(s[f] for s in states)) for f in states[0]},
            model_loaded=False,
            sealed_audit_loaded=False,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha(args.output / "result.json"),
                states=len(states),
                conditions=len(data),
            )
        )
    )


if __name__ == "__main__":
    main()
