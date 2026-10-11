"""Fixed-budget equal-source final-block repair experiment; not exact resume."""

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def canonical_state(text):
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except (ValueError, TypeError):
        return text.strip()


def verify_splits(train, validation):
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("ID overlap")
    if {canonical_state(r["state"]) for r in train} & {
        canonical_state(r["state"]) for r in validation
    }:
        raise ValueError("state overlap")
    for rows in (train, validation):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate IDs")


def schedule(sources, steps, seed=17):
    rng = random.Random(seed)
    orders = []
    for rows in sources:
        order = list(range(len(rows)))
        rng.shuffle(order)
        if len(order) < steps * 4:
            raise ValueError("insufficient unique examples")
        orders.append(order)
    return [
        [(s, i) for s in range(2) for i in orders[s][step * 4 : (step + 1) * 4]]
        for step in range(steps)
    ]


def configure_repair_policy(model, policy):
    if policy not in ("last_block", "all_encoder_blocks"):
        raise ValueError("unknown training policy")
    prefix = "bert.encoder.layer.5." if policy == "last_block" else "bert.encoder.layer."
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith((prefix, "classifier."))
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise ValueError("no matching encoder parameters")
    return names


def curriculum_training_contract(root, policy):
    if policy == "original":
        return root / "conditional-curriculum-01/train.jsonl", (
            "4d6110f4a401cfd6e50b9d0720a3a5fcc3aea146118ff7d1b0a05351764afbbe"
        )
    if policy == "numeric_range":
        return root / "numeric-range-curriculum-01/train.jsonl", (
            "95f4edc33711a6ac334fff77cb08300a45bbe2bc8acefad1e8591b069c076101"
        )
    raise ValueError("unknown curriculum policy")


def main():
    import torch
    import yaml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_frozen_text_probe import _load_examples, _metrics
    from scripts.train_text_crossencoder import choice_loss, configure_trainable, frozen_hash
    from tiny_omni_decision.dataset import sha256_file
    from tiny_omni_decision.schema import DatasetManifest

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, choices=(512, 2048), default=512)
    parser.add_argument(
        "--trainable-policy", choices=("last_block", "all_encoder_blocks"), default="last_block"
    )
    parser.add_argument(
        "--curriculum-training", choices=("original", "numeric_range"), default="original"
    )
    args = parser.parse_args()
    curriculum_train_path, curriculum_train_hash = curriculum_training_contract(
        args.root, args.curriculum_training
    )
    paths = {
        args.root / "text/train.jsonl": (
            "1f3e0f751b6fdd1115cf6af4ea3ebc13545a1148614eda4f9f56ae8492c80675"
        ),
        args.root / "text/validation.jsonl": (
            "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8"
        ),
        curriculum_train_path: curriculum_train_hash,
        args.root / "conditional-curriculum-01/validation.jsonl": (
            "6ff01e9af03cf481e875dec643e7824ef9fc61718337c7c18774484dea30a34c"
        ),
        args.root / "text-crossencoder-adapt-v1/best-delta.pt": (
            "5fa8323a47e72c3796d0da35618b603fa49784cacb670dee0e910e8d5757ed78"
        ),
        args.model / "model.safetensors": (
            "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
        ),
    }
    for path, expected in paths.items():
        if sha256_file(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )

    def original(split):
        rows = _load_examples(
            args.root / f"text/{split}.jsonl", manifest.model_copy(update={"split": split}), True
        )
        kinds = {}
        for line in (args.root / f"text/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            for key, value in json.loads(raw["questions"]).items():
                kinds[f"{raw['state_id']}:{key}"] = value["type"]
        return [
            dict(
                id=e.id,
                state=e.state,
                question=e.question,
                options=e.options,
                target=e.target,
                group=e.source_record_id.split(":")[0],
                task=kinds[e.source_record_id],
                source="original",
            )
            for e in rows
        ]

    def curriculum(split):
        path = (
            curriculum_train_path
            if split == "train"
            else (args.root / "conditional-curriculum-01/validation.jsonl")
        )
        return [
            dict(json.loads(line), source="curriculum")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    trains = [original("train"), curriculum("train")]
    validations = [original("validation"), curriculum("validation")]
    verify_splits(sum(trains, []), sum(validations, []))
    steps = 2 if args.smoke else args.steps
    if args.smoke:
        validations = [r[:8] for r in validations]
    planned = schedule(trains, steps)
    train = [trains[source][index] for batch in planned for source, index in batch]
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(
        "config.json",
        {
            "argv": sys.argv,
            "started_utc": datetime.now(UTC).isoformat(),
            "seed": 17,
            "trainable_policy": args.trainable_policy,
            "curriculum_training": args.curriculum_training,
            "steps": steps,
            "smoke": args.smoke,
            "batch": 8,
            "lr": 2e-5,
            "weight_decay": 0.01,
            "clip": 1.0,
            "dropout": False,
            "optimizer_reset": True,
            "initialization": "saved adapted epoch3 weights; new AdamW",
            "selector": "earliest minimum mean of original and curriculum NLL, including step0",
            "paths": {str(p): h for p, h in paths.items()},
            "runner_sha256": sha256_file(Path(__file__)),
            "train_ids": [r["id"] for r in train],
            "validation_ids": [[r["id"] for r in rows] for rows in validations],
        },
    )
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def encode(rows):
        encoded = []
        for row in rows:
            first = [f"Question: {row['question']}\nCandidate answer: {o}" for o in row["options"]]
            if max(map(len, tokenizer(first, add_special_tokens=False)["input_ids"])) + 3 >= 512:
                raise ValueError("question/option overflow")
            values = tokenizer(
                first,
                [f"State: {row['state']}"] * len(first),
                max_length=512,
                truncation="only_second",
            )
            encoded.append([{k: v[i] for k, v in values.items()} for i in range(len(first))])
        return encoded

    train_encoded = encode(train)
    val_encoded = [encode(rows) for rows in validations]
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        .to(device)
        .eval()
    )
    names = configure_trainable(model)
    delta = torch.load(
        args.root / "text-crossencoder-adapt-v1/best-delta.pt",
        weights_only=True,
        map_location="cpu",
    )
    if set(names) != set(delta):
        raise ValueError("initial delta names mismatch")
    model.load_state_dict(delta, strict=False)
    names = configure_repair_policy(model, args.trainable_policy)
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in parameters)
    expected_count = 1774849 if args.trainable_policy == "last_block" else 10647169
    if trainable_count != expected_count:
        raise ValueError("trainable parameter mismatch")
    write(
        "trainable-parameters.json",
        {
            "policy": args.trainable_policy,
            "names": names,
            "trainable_count": trainable_count,
            "total_count": sum(p.numel() for p in model.parameters()),
        },
    )
    before = frozen_hash(model)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)

    def forward(encoded):
        inputs = tokenizer.pad(
            [v for group in encoded for v in group], padding=True, return_tensors="pt"
        ).to(device)
        scores = model(**inputs).logits[:, 0]
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite logits")
        return scores, [len(group) for group in encoded]

    def snapshot():
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    started = time.perf_counter()
    history = []
    best = float("inf")
    best_step = 0
    best_predictions = None
    total_ce = 0.0
    last_norm = None

    def evaluate(step):
        nonlocal best, best_step, best_predictions
        stamp = time.perf_counter()
        all_values = []
        groups = {}
        all_metrics = []
        with torch.inference_mode():
            for rows, encoded in zip(validations, val_encoded, strict=True):
                values = []
                for start in range(0, len(rows), 8):
                    scores, widths = forward(encoded[start : start + 8])
                    values.extend(v.cpu() for v in scores.split(widths))
                metrics = _metrics(values, [r["options"].index(r["target"]) for r in rows])
                all_metrics.append(metrics)
                for task in sorted({r["task"] for r in rows}):
                    indices = [i for i, r in enumerate(rows) if r["task"] == task]
                    groups[rows[0]["source"] + ":" + task] = _metrics(
                        [values[i] for i in indices],
                        [rows[i]["options"].index(rows[i]["target"]) for i in indices],
                    )
                write(
                    f"predictions-{step}-{rows[0]['source']}.json",
                    [
                        dict(
                            id=r["id"],
                            target=r["target"],
                            options=r["options"],
                            probabilities=v.softmax(-1).tolist(),
                        )
                        for r, v in zip(rows, values, strict=True)
                    ],
                )
                all_values.append(values)
        if frozen_hash(model) != before:
            raise ValueError("frozen weights changed")
        score = sum(m["nll"] for m in all_metrics) / 2
        if score < best:
            best, best_step = score, step
            best_predictions = all_values
            torch.save(snapshot(), args.output / "best-delta.pt")
        used = train[: step * 8]
        record = {
            "step": step,
            "selector": score,
            "best_step": best_step,
            "original": all_metrics[0],
            "curriculum": all_metrics[1],
            "tasks": groups,
            "rolling_train_ce": total_ce / step if step else None,
            "last_gradient_norm": last_norm,
            "consumed": len(used),
            "unique_questions": len({r["id"] for r in used}),
            "unique_states": len({r["group"] for r in used}),
            "source_counts": dict(Counter(r["source"] for r in used)),
            "task_counts": dict(Counter(r["source"] + ":" + r["task"] for r in used)),
            "evaluation_seconds": time.perf_counter() - stamp,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        write("history.json", history)
        print(
            json.dumps(
                {k: record[k] for k in ("step", "selector", "best_step", "elapsed_seconds")}
            ),
            flush=True,
        )

    evaluate(0)
    if not args.smoke:
        reference = json.loads(
            (args.root / "text-crossencoder-adapt-v1/result.json").read_text(encoding="utf-8")
        )["metrics"]
        for key in ("accuracy", "nll", "brier", "ece_15_bins"):
            if abs(history[0]["original"][key] - reference[key]) > 1e-6:
                raise ValueError("initial original-validation reference mismatch")
    for step in range(1, steps + 1):
        start = (step - 1) * 8
        scores, widths = forward(train_encoded[start : start + 8])
        targets = torch.tensor(
            [r["options"].index(r["target"]) for r in train[start : start + 8]], device=device
        )
        loss = choice_loss(scores, widths, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("frozen gradient")
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters):
            raise ValueError("no trainable gradient")
        optimizer.step()
        total_ce += float(loss.detach())
        last_norm = float(norm)
        if step % 128 == 0 or step == steps:
            evaluate(step)
    torch.save(snapshot(), args.output / "final-delta.pt")
    model.load_state_dict(
        torch.load(args.output / "best-delta.pt", weights_only=True), strict=False
    )
    with torch.inference_mode():
        for encoded, expected in zip(val_encoded, best_predictions, strict=True):
            actual = []
            for start in range(0, len(encoded), 8):
                scores, widths = forward(encoded[start : start + 8])
                actual.extend(v.cpu() for v in scores.split(widths))
            if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise ValueError("best reload mismatch")
    write(
        "result.json",
        {
            "status": "PASS",
            "smoke": args.smoke,
            "steps": steps,
            "best_step": best_step,
            "frozen_sha256": before,
            "best_sha256": sha256_file(args.output / "best-delta.pt"),
            "final_sha256": sha256_file(args.output / "final-delta.pt"),
            "seconds": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "trainable_parameters": trainable_count,
            "trainable_policy": args.trainable_policy,
            "exact_resume": False,
        },
    )


if __name__ == "__main__":
    main()
