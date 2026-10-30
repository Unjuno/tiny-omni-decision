"""Saved-prediction, posthoc development retention audit; no model loading."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.compare_conditional_runs import paired_accuracy


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.root / "conditional-repair-range-warmstart-01"
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    config = read(run / "config.json")
    result = read(run / "result.json")
    history = read(run / "history.json")
    if result["status"] != "PASS" or result["steps"] != 2048:
        raise ValueError("incomplete run")
    if [r["step"] for r in history] != list(range(0, 2049, 128)):
        raise ValueError("incomplete history")
    selected = min(history, key=lambda r: r["selector"])["step"]
    if selected != result["best_step"]:
        raise ValueError("selector mismatch")
    corpus = args.root / "conditional-curriculum-01/validation.jsonl"
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if config["paths"].get(str(corpus)) != digest:
        raise ValueError("corpus hash mismatch")
    hashes[str(corpus)] = digest
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    before = read(run / "predictions-0-curriculum.json")
    comparisons = {}
    for label, step in (("selected", selected), ("final", 2048)):
        after = read(run / f"predictions-{step}-curriculum.json")
        paired_accuracy(rows, before, after)  # Validate the full IDs/order first.
        groups = {}
        for task in ["overall", *sorted({r["task"] for r in rows})]:
            ids = [i for i, r in enumerate(rows) if task == "overall" or r["task"] == task]
            differences = np.array(
                paired_accuracy(*[[source[i] for i in ids] for source in (rows, before, after)])
            )
            rng = np.random.default_rng(20261012)
            draws = differences[
                rng.integers(0, len(differences), (10000, len(differences)))
            ].mean(1)
            groups[task] = {
                "questions": len(ids),
                "states": len(differences),
                "accuracy_difference": float(differences.mean()),
                "percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
        comparisons[label] = {"step": step, "groups": groups, "metrics": history[step // 128]}
    report = {
        "initial_metrics": history[0],
        "comparisons": comparisons,
        "input_sha256": hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_seed": 20261012,
        "replicates": 10000,
        "limitation": "Posthoc reused development validation; selection and multiple comparisons "
        "are not corrected. No new transfer, blind audit or causal attribution.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
