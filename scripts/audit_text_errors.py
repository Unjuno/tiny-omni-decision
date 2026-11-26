"""Read-only error attribution on saved development predictions; no new inference."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def teacher_distribution(kind, options, teacher):
    try:
        if kind == "noul":
            value = float(teacher["noul"])
            probabilities = [value, 1 - value]
        else:
            keys = (
                [o.split(": ", 1)[0] for o in options]
                if kind == "choice"
                else [str(i) for i in range(len(options))]
            )
            probabilities = [float(teacher["probabilities"][k]) for k in keys]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing or malformed teacher probabilities") from exc
    if (
        len(probabilities) != len(options)
        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
        or abs(sum(probabilities) - 1) > 0.001
    ):
        raise ValueError("invalid teacher distribution")
    return probabilities


def summarize(rows):
    if not rows:
        return {"count": 0}
    loss = brier = correct = 0
    bins = defaultdict(list)
    for row in rows:
        p, target = row["probabilities"], row["target_index"]
        predicted = max(range(len(p)), key=p.__getitem__)
        hit = int(predicted == target)
        correct += hit
        loss -= math.log(max(p[target], 1e-12))
        brier += sum((v - int(i == target)) ** 2 for i, v in enumerate(p))
        bins[min(14, int(max(p) * 15))].append((max(p), hit))
    n = len(rows)
    ece = sum(abs(sum(c - h for c, h in values)) for values in bins.values()) / n
    return dict(
        count=n,
        errors=n - correct,
        accuracy=correct / n,
        nll=loss / n,
        brier=brier / n,
        ece=ece,
        ece_bins=15,
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import yaml
    from transformers import AutoTokenizer

    from scripts.train_frozen_text_probe import _load_examples
    from tiny_omni_decision.schema import DatasetManifest

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "text/validation.jsonl"
    predictions = root / "numeric-deterministic-control-01/predictions-1664-original.json"
    if sha(source) != "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8":
        raise ValueError("corpus changed")
    out = root / "text-error-attribution-01"
    out.mkdir(exist_ok=False)
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )
    examples = _load_examples(source, manifest.model_copy(update={"split": "validation"}), True)
    saved = json.loads(predictions.read_text(encoding="utf-8"))
    if [r["id"] for r in saved] != [e.id for e in examples]:
        raise ValueError("prediction ID/order mismatch")
    definitions = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key, definition in json.loads(row["questions"]).items():
            definitions[f"{row['state_id']}:{key}"] = (
                definition,
                json.loads(row["teacher"])[key],
                row["domain"],
            )
    tokenizer_path = Path(
        "C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    records, groups = [], defaultdict(list)
    for example, prediction in zip(examples, saved, strict=True):
        if prediction["options"] != example.options or prediction["target"] != example.target:
            raise ValueError("option/target mismatch")
        definition, teacher, domain = definitions[example.source_record_id]
        first = [f"Question: {example.question}\nCandidate answer: {o}" for o in example.options]
        second = [f"State: {example.state}"] * len(first)
        lengths = [len(ids) for ids in tokenizer(first, second, truncation=False)["input_ids"]]
        probs = prediction["probabilities"]
        if (
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs)
            or abs(sum(probs) - 1) > 1e-5
        ):
            raise ValueError("invalid saved probabilities")
        target = example.options.index(example.target)
        tp = teacher_distribution(definition["type"], example.options, teacher)
        teacher_top = max(range(len(tp)), key=tp.__getitem__)
        row = dict(
            id=example.id,
            target_index=target,
            probabilities=probs,
            question_type=definition["type"],
            domain=domain,
            max_pair_tokens=max(lengths),
            truncated_pairs=sum(v > 512 for v in lengths),
            teacher_top_matches_gold=teacher_top == target,
            teacher_gold_probability=tp[target],
            teacher_probabilities=tp,
        )
        records.append(row)
        categories = [
            "all",
            "type:" + definition["type"],
            "domain:" + domain,
            "context:" + ("truncated" if row["truncated_pairs"] else "complete"),
            "teacher:" + ("agrees" if teacher_top == target else "disagrees"),
            "teacher_gold:" + ("below_0.8" if tp[target] < 0.8 else "at_least_0.8"),
        ]
        for category in categories:
            groups[category].append(row)
    metrics = {k: summarize(v) for k, v in groups.items()}
    reference = summarize(
        [
            dict(probabilities=r["teacher_probabilities"], target_index=r["target_index"])
            for r in records
        ]
    )
    payload = dict(
        status="PASS_DIAGNOSTIC_ONLY",
        checkpoint_step=1664,
        corpus_sha256=sha(source),
        predictions_sha256=sha(predictions),
        runner_sha256=sha(Path(__file__)),
        groups=metrics,
        teacher_reference=reference,
        limitations=[
            "Observed checkpoint-selection validation, not blind evaluation",
            "Teacher disagreement is not proof of label error or a quality ceiling",
            "Associations across subgroups are not causal effects",
            "No filtering, reweighting, inference, training or calibration",
        ],
    )
    for name, data in [("result.json", payload), ("records.json", records)]:
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    doc = Path("docs/pretrained_reuse/R2_TEXT_ERROR_ATTRIBUTION_20261012")
    doc.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Saved text prediction error attribution",
        "",
        "All2540 original validation questions retained; selected control step1664.",
        "Read-only diagnostic; no new inference, training, data filtering or checkpoint selection.",
        "",
        "| Group | Count | Errors | Accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, m in metrics.items():
        lines.append(
            f"| {k} | {m['count']} | {m['errors']} | {m['accuracy']:.6f} "
            f"| {m['nll']:.6f} | {m['brier']:.6f} | {m['ece']:.6f} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        *["- " + v for v in payload["limitations"]],
        f"- Full per-ID attribution and input hashes: `{out.as_posix()}`.",
        "- Teacher distributions are upstream metadata, not a newly measured local model.",
    ]
    doc.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("domain:")}))


if __name__ == "__main__":
    main()
