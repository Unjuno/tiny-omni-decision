"""Frozen five-arm, two-checkpoint development comparison; no training."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.build_conditional_curriculum import build, verify

VARIANTS = ("control", "interpolation", "extrapolation", "wording", "combined")
CHECKPOINTS = {
    "parent": ("conditional-repair-encoder-coverage-01",
               "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318"),
    "candidate": ("conditional-repair-range-warmstart-01",
                  "f287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content(row):
    return json.dumps([json.loads(row["state"]), row["question"].split(), row["options"]],
                      sort_keys=True)


def transform(row, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    row = copy.deepcopy(row)
    scale, offset = {"interpolation": (7, 19), "extrapolation": (13, 29),
                     "combined": (13, 29)}.get(variant, (1, 0))
    state = json.loads(row["state"])
    for item in state:
        for field in ("cost", "risk", "time"):
            item[field] = scale * item[field] + offset
    if "threshold" in row["oracle"]:
        old = row["oracle"]["threshold"]
        row["oracle"]["threshold"] = scale * old + offset
        row["question"] = row["question"].replace(str(old), str(row["oracle"]["threshold"]))
    if variant in ("wording", "combined"):
        oracle = row["oracle"]
        if row["task"] in ("extremum", "temporal_order"):
            direction = "smallest" if oracle["direction"] == "minimum" else "largest"
            row["question"] = f"Which item has the {direction} {oracle['field']}?"
        elif row["task"] == "count":
            row["question"] = (
                "Return the number of items with risk less than or equal to "
                f"{oracle['threshold']}."
            )
        else:
            row["question"] = (
                f"Exclude items whose risk exceeds {oracle['threshold']}. Of the remaining "
                "items, which has the smallest cost? Answer none if no items remain."
            )
    row["state"] = json.dumps(state)
    verify(row)
    row.update(base_id=row["id"], id=variant + ":" + row["id"], variant=variant)
    return row


def reject_overlap(rows, previous):
    if {r["group"] for r in rows} & {r["group"] for r in previous}:
        raise ValueError("latent state overlap")
    if {content(r) for r in rows} & {content(r) for r in previous}:
        raise ValueError("normalized content overlap")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate IDs")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(root, output):
    rows = [transform(r, v) for v in VARIANTS for r in build(97, 256, "development")]
    paths = [root / name for name in (
        "conditional-curriculum-01/train.jsonl", "conditional-curriculum-01/validation.jsonl",
        "numeric-range-curriculum-01/train.jsonl", "numeric-range-curriculum-01/transfer.jsonl",
        "condition-transfer-controls-01/inputs.jsonl",
    )]
    previous = [json.loads(line) for p in paths
                for line in p.read_text(encoding="utf-8").splitlines()]
    reject_overlap(rows, previous)
    output.mkdir(parents=True, exist_ok=False)
    data = output / "inputs.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = {}
    for variant in VARIANTS:
        states = {r["group"]: json.loads(r["state"]) for r in rows if r["variant"] == variant}
        values = [item[f] for state in states.values() for item in state
                  for f in ("cost", "risk", "time")]
        counts[variant] = dict(states=len(states), numeric_values=len(values),
                               above_training_max=sum(v > 1592 for v in values))
    write(output / "manifest.json", dict(seed=97, states=256, rows=len(rows),
          input_sha256=sha(data), previous_sha256={str(p): sha(p) for p in paths},
          counts=counts, builder_sha256=sha(Path(__file__)), overlap=0))


def evaluate(root, model_path, data, expected_input, output, checkpoints=None):
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.compare_conditional_runs import paired_accuracy
    from scripts.train_frozen_text_probe import _metrics

    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    if set(checkpoints) != {"parent", "candidate"}:
        raise ValueError("expected fixed parent/candidate comparison")
    inputs = data / "inputs.jsonl"
    if sha(inputs) != expected_input:
        raise ValueError("input hash mismatch")
    base_hash = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    if sha(model_path / "model.safetensors") != base_hash:
        raise ValueError("base mismatch")
    for name, digest in checkpoints.values():
        if sha(root / name / "best-delta.pt") != digest:
            raise ValueError("checkpoint mismatch")
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 10240 or len({r["id"] for r in rows}) != 10240:
        raise ValueError("incomplete inputs")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", dict(input_sha256=expected_input, checkpoints=checkpoints,
          base_sha256=base_hash, runner_sha256=sha(Path(__file__)), training=False,
          selection=False, seed=20261012, resamples=10000))
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    reports, predictions = {}, {}
    for name, (directory, _) in checkpoints.items():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True).to(device).eval()
        delta = torch.load(root / directory / "best-delta.pt", weights_only=True,
                           map_location="cpu")
        if set(delta) != {n for n, _ in model.named_parameters()
                          if n.startswith(("bert.encoder.layer.", "classifier."))}:
            raise ValueError("delta scope mismatch")
        model.load_state_dict(delta, strict=False)
        model.requires_grad_(False)
        reports[name], predictions[name] = {}, {}
        with torch.inference_mode():
            for variant in VARIANTS:
                selected = [r for r in rows if r["variant"] == variant]
                values, max_tokens = [], 0
                for offset in range(0, len(selected), 8):
                    batch = selected[offset:offset + 8]
                    first = [f"Question: {r['question']}\nCandidate answer: {o}"
                             for r in batch for o in r["options"]]
                    second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
                    tokens = tokenizer(first, second, padding=True, truncation=False,
                                       return_tensors="pt")
                    max_tokens = max(max_tokens, tokens["input_ids"].shape[1])
                    if max_tokens > 512:
                        raise ValueError("input would truncate")
                    logits = model(**tokens.to(device)).logits[:, 0].cpu()
                    if not torch.isfinite(logits).all():
                        raise ValueError("nonfinite logits")
                    values.extend(logits.split([len(r["options"]) for r in batch]))
                targets = [r["options"].index(r["target"]) for r in selected]
                metrics = {"overall": _metrics(values, targets), "tasks": {},
                           "max_tokens": max_tokens}
                for task in sorted({r["task"] for r in selected}):
                    ids = [i for i, r in enumerate(selected) if r["task"] == task]
                    metrics["tasks"][task] = _metrics([values[i] for i in ids],
                                                       [targets[i] for i in ids])
                pred = [dict(id=r["id"], options=r["options"], target=r["target"],
                             probabilities=v.softmax(-1).tolist())
                        for r, v in zip(selected, values, strict=True)]
                write(output / f"{name}-{variant}-predictions.json", pred)
                predictions[name][variant] = pred
                reports[name][variant] = metrics
                print(name, variant, metrics["overall"], flush=True)
        del model, delta
        if device == "cuda":
            torch.cuda.empty_cache()
    comparisons = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        diff = np.array(paired_accuracy(selected, predictions["parent"][variant],
                                        predictions["candidate"][variant]))
        rng = np.random.default_rng(20261012)
        draws = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
        comparisons[variant] = dict(accuracy_difference=float(diff.mean()),
                percentile_95_interval=np.quantile(draws, [0.025, 0.975]).tolist())
    write(output / "result.json", dict(status="PASS_EXECUTION_ONLY", metrics=reports,
          comparisons=comparisons, seconds=time.perf_counter()-start, device=device,
          peak_vram_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
          limitation="Development comparison, unequal prior compute; no blind or causal claim."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "evaluate"])
    for name in ("root", "output", "model", "data"):
        p.add_argument("--" + name, type=Path, required=name in ("root", "output"))
    p.add_argument("--input-sha256")
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.output)
    else:
        if not all((args.model, args.data, args.input_sha256)):
            p.error("evaluate requires model, data and input-sha256")
        evaluate(args.root, args.model, args.data, args.input_sha256, args.output)


if __name__ == "__main__":
    main()
