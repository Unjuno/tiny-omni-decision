"""Recompute saved development metrics without model inference or selection."""

import argparse
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.audit_native_training_pipeline import REPO, ROOT, read, sha, write
from scripts.native_pipeline_audit import independent_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started, pins = time.perf_counter(), {}

    def verify(path, digest=None):
        actual = sha(path)
        if digest is not None and digest != actual:
            raise ValueError("changed frozen metric input: " + str(path))
        pins[str(path)] = actual

    verify(Path(__file__))
    verify(REPO / "scripts/native_pipeline_audit.py")
    corpus, fit = ROOT / "native-common-corpus-02", ROOT / "native-long-budget-fit-01"
    verify(corpus / "result.json",
           "47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49")
    verify(fit / "result.json",
           "645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b")
    receipt = read(fit / "result.json")
    verify(corpus / "validation.jsonl",
           read(corpus / "result.json")["artifacts"]["validation.jsonl"])
    for name in ("validation-ids.json", "learning-curve.json"):
        verify(fit / name, receipt["artifacts"][name])
    rows = read(corpus / "validation.jsonl")
    if read(fit / "validation-ids.json") != [
        {k: r[k] for k in ("id", "option_ids", "target_index")} for r in rows
    ]:
        raise ValueError("saved prediction-to-target association differs")
    target = np.array([r["target_index"] for r in rows])
    groups = {name: defaultdict(list) for name in ("tasks", "sources", "modalities")}
    for i, row in enumerate(rows):
        groups["tasks"][row["source"] + ":" + row["task"]].append(i)
        groups["sources"][row["source"]].append(i)
        groups["modalities"][row["modality"]].append(i)
    names = ("accuracy", "nll", "brier", "ece")
    output, comparisons, max_delta = [], 0, 0.0

    def compare(current, expected):
        nonlocal comparisons, max_delta
        for k, value in current.items():
            if isinstance(value, dict):
                compare(value, expected[k])
            else:
                delta = abs(value - expected[k])
                if delta > 1e-10:
                    raise ValueError("saved metric differs: " + k)
                comparisons += 1
                max_delta = max(max_delta, delta)

    for point in read(fit / "learning-curve.json"):
        step = point["step"]
        name = f"validation-{step:04}.npy"
        verify(fit / name, receipt["artifacts"][name])
        logits = np.load(fit / name, allow_pickle=False)
        if logits.shape != (len(rows), 40):
            raise ValueError("complete padded candidate scores required")
        for i, row in enumerate(rows):
            if np.any(logits[i, len(row["options"]):] != -1e30):
                raise ValueError("noncandidate score contributes to decision")
        values = {scope: {key: independent_metrics(logits[index], target[index])
                          for key, index in lookup.items()} for scope, lookup in groups.items()}
        values["overall"] = independent_metrics(logits, target)
        values["source_macro"] = {
            source: {k: float(np.mean([values["tasks"][source + ":" + task][k] for task in
                     sorted({rows[i]["task"] for i in index})])) for k in names}
            for source, index in groups["sources"].items()}
        values["modality_macro"] = {
            modality: {k: float(np.mean([values["source_macro"][s][k] for s in
                       sorted({rows[i]["source"] for i in index})])) for k in names}
            for modality, index in groups["modalities"].items()}
        values["macro"] = {k: float(np.mean([m[k] for m in values["modality_macro"].values()]))
                           for k in names}
        compare(values, point["validation"])
        ties = int(((logits == logits.max(1)[:, None]).sum(1) > 1).sum())
        output.append(dict(step=step, exact_top_score_ties=ties, macro=values["macro"]))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError("metric audit input changed")
    report = dict(status="PASS_INDEPENDENT_SAVED_METRICS", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        validation_rows=len(rows), checkpoints=output, scalar_comparisons=comparisons,
        maximum_absolute_error=max_delta, validated_hashes=pins,
        seconds=time.perf_counter() - started, model_inference=False, optimizer_updates=0,
        checkpoint_selection=False, sealed_audit_loaded=False)
    write(args.output / "result.json", report)
    print(report["status"], comparisons, max_delta)


if __name__ == "__main__":
    main()
