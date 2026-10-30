"""Summarize pinned saved predictions and TRAIN wording coverage, without inference."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.primary / "result.json") != args.primary_sha256:
        raise ValueError("primary hash mismatch")
    result = read(args.primary / "result.json")
    if result["status"] != "PASS_FIXED_SEMANTIC_NATIVE_COMPARISON" or result["smoke"]:
        raise ValueError("completed full comparison required")
    pins = {str(args.primary / "result.json"): args.primary_sha256}
    for name, digest in result["artifacts"].items():
        if sha(args.primary / name) != digest:
            raise ValueError("saved payload changed: " + name)
        pins[str(args.primary / name)] = digest
    args.output.mkdir(parents=True, exist_ok=False)
    states, stored = read(args.primary / "states.json"), read(args.primary / "metrics.json")
    errors, rows_csv = {}, []
    for arm in ("step0", "endpoint"):
        for kind in ("oracle", "predicted"):
            rows = [
                json.loads(line)
                for line in (args.primary / f"{arm}-{kind}-predictions.jsonl")
                .read_text()
                .splitlines()
            ]
            if len(rows) != 1024:
                raise ValueError("incomplete predictions")
            confusion, attribution = Counter(), Counter()
            for i, row in enumerate(rows):
                if row["prediction"] == row["target"]:
                    continue
                requested = row["option_ids"][row["target"]].split(":", 1)[0]
                answered, value = row["predicted_id"].split(":", 1)
                confusion[f"{requested}->{answered}"] += 1
                correct_fact = states[kind][i // 4][answered] == value
                attribution[
                    "true_but_unrequested" if requested != answered and correct_fact else "other"
                ] += 1
            errors[f"{arm}:{kind}"] = dict(
                count=sum(confusion.values()),
                confusion=dict(confusion),
                attribution=dict(attribution),
            )
        for kind, values in stored["arms"][arm].items():
            if not isinstance(values, dict) or "overall" not in values:
                continue
            for name, value in {"overall": values["overall"], **values["per_question"]}.items():
                rows_csv.append(dict(arm=arm, condition=kind, task=name, **value))
    corpus = Path(read(args.primary / "effective-config.json")["config"]["artifact_root"])
    corpus /= "decision-supervision-corpus-01"
    coverage = {}
    for name, expected in (
        ("train.jsonl", "ecadf350f1003842703a5f41aeb2eb5ab760b236130c4d1f0543b9b1886a53b3"),
        (
            "retention_train.jsonl",
            "a6d363fe4349432ecaf2a29465ee166e7f53ddf577d930884afaa0327df0814c",
        ),
    ):
        path = corpus / name
        if sha(path) != expected:
            raise ValueError("frozen TRAIN coverage source changed")
        pins[str(path)] = expected
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        questions = Counter(row["question"] for row in rows)
        coverage[name] = dict(
            rows=len(rows),
            exact_native_count_question=questions["How many objects are there?"],
            how_many_rows=sum(n for q, n in questions.items() if "how many" in q.lower()),
        )
    out = dict(
        status="PASS_SAVED_NATIVE_ERROR_ACCOUNTING",
        errors=errors,
        train_wording_coverage=coverage,
        input_pins=pins,
        script_sha256=sha(Path(__file__)),
        interpretation="Observed interface-transfer failures; coverage is not causal proof.",
        transfer_loaded=False,
        sealed_audit_loaded=False,
        optimizer_updates=0,
    )
    (args.output / "error-accounting.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per-task-metrics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    print(json.dumps(dict(status=out["status"], metric_rows=len(rows_csv), errors=errors)))


if __name__ == "__main__":
    main()
