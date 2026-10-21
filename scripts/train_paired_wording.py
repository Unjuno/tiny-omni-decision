"""Matched wording-view training, isolated from historical runs."""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def check_order(order, lookup, count):
    if len(order) != count or len(set(order)) != count or not set(order) <= set(lookup):
        raise ValueError("invalid training order")


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import transformers
    import yaml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.range_generalization import sha
    from scripts.train_frozen_text_probe import _load_examples, _metrics
    from scripts.train_numeric_deterministic import configure_repair_policy, verify_splits
    from scripts.train_text_crossencoder import choice_loss, frozen_hash
    from tiny_omni_decision.schema import DatasetManifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("control", "wording"), required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    preflight = json.loads(
        (root / "text-coverage-preflight-01/result.json").read_text(encoding="utf-8")
    )
    for p, h in preflight["input_hashes"].items():
        if sha(Path(p)) != h:
            raise ValueError("input changed")
    checkpoint = root / "text-coverage-training-01/best-delta.pt"
    if sha(checkpoint) != "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5":
        raise ValueError("checkpoint changed")
    if (
        sha(model_path / "model.safetensors")
        != "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base changed")
    view_path = root / "paired-wording-training-views-02/train-views.jsonl"
    if sha(view_path) != "3e7722c5bd8494aa5b6f714b55f40f2ee551f82014007687bdb543a3977bb3de":
        raise ValueError("view corpus changed")
    if shutil.disk_usage(root).free < 704643072:
        raise ValueError("disk preflight failed")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )

    def original(split):
        p = root / f"text/{split}.jsonl"
        kinds = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            for k, d in json.loads(r["questions"]).items():
                kinds[r["state_id"] + ":" + k] = d["type"]
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
            for e in _load_examples(p, manifest.model_copy(update={"split": split}), True)
        ]

    def curriculum(path):
        return [
            dict(json.loads(line), source="curriculum")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    pool = original("train") + curriculum(root / "comparison-vocabulary-curriculum-01/train.jsonl")
    validations = [
        original("validation"),
        curriculum(root / "conditional-curriculum-01/validation.jsonl"),
    ]
    verify_splits(pool, sum(validations, []))
    from scripts.paired_wording_schedule import schedule

    old_config = json.loads(
        (root / "text-coverage-training-01/config.json").read_text(encoding="utf-8")
    )
    if [[r["id"] for r in rows] for rows in validations] != old_config["validation_ids"]:
        raise ValueError("validation order changed")
    views = curriculum(view_path)
    train = schedule(pool, views, old_config["train_ids"], args.arm)
    steps = 2 if args.smoke else 1024
    if args.smoke:
        train = train[:8] + train[2048:2056]
        validations = [v[:8] for v in validations]
    args.output.mkdir(exist_ok=False, parents=True)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def encode(rows):
        encoded = []
        for r in rows:
            first = [f"Question: {r['question']}\nCandidate answer: {o}" for o in r["options"]]
            if max(map(len, tokenizer(first, add_special_tokens=False)["input_ids"])) + 3 >= 512:
                raise ValueError("question overflow")
            b = tokenizer(
                first,
                [f"State: {r['state']}"] * len(first),
                max_length=512,
                truncation="only_second",
            )
            encoded.append([{k: v[i] for k, v in b.items()} for i in range(len(first))])
        return encoded

    encoded_train = encode(train)
    encoded_val = [encode(v) for v in validations]
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        .to(device)
        .eval()
    )
    names = configure_repair_policy(model, "all_encoder_blocks")
    delta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(delta) != set(names):
        raise ValueError("checkpoint names differ")
    model.load_state_dict(delta, strict=False)
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 10647169:
        raise ValueError("parameter count")
    before = frozen_hash(model)
    write(
        "config.json",
        dict(
            argv=sys.argv,
            started_utc=datetime.now(UTC).isoformat(),
            smoke=args.smoke,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            seed=17,
            steps=steps,
            batch=8,
            lr=2e-5,
            weight_decay=0.01,
            clip=1,
            dropout=False,
            deterministic=True,
            optimizer_reset=True,
            exact_resume=False,
            initialization=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            arm=args.arm,
            view_corpus_sha256=sha(view_path),
            parent_ids=[r.get("parent_id", r["id"]) for r in train],
            input_hashes=preflight["input_hashes"],
            train_ids=[r["id"] for r in train],
            validation_ids=[[r["id"] for r in v] for v in validations],
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            attention_backend=model.config._attn_implementation,
            trainable_parameters=10647169,
            total_parameters=sum(p.numel() for p in model.parameters()),
            selector="earliest minimum equal-source mean NLL, including step0",
        ),
    )
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)

    def forward(groups):
        batch = tokenizer.pad([x for g in groups for x in g], return_tensors="pt", padding=True).to(
            device
        )
        scores = model(**batch).logits[:, 0]
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite scores")
        return scores, [len(g) for g in groups]

    def save(name):
        if shutil.disk_usage(root).free < 536870912 + 43 * 1024**2:
            raise ValueError("disk reserve")
        temporary = args.output / (name + ".tmp")
        torch.save(
            {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
            temporary,
        )
        temporary.replace(args.output / name)

    history = []
    best = float("inf")
    best_step = 0
    best_values = None
    total_loss = 0.0
    norm = None
    start = time.perf_counter()

    def evaluate(step):
        nonlocal best, best_step, best_values
        tick = time.perf_counter()
        all_values = []
        metrics = {}
        tasks = {}
        with torch.inference_mode():
            for rows, encoded in zip(validations, encoded_val, strict=True):
                values = []
                for i in range(0, len(rows), 8):
                    scores, widths = forward(encoded[i : i + 8])
                    values.extend(x.cpu() for x in scores.split(widths))
                source = rows[0]["source"]
                all_values.append(values)
                metrics[source] = _metrics(values, [r["options"].index(r["target"]) for r in rows])
                for task in sorted({r["task"] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r["task"] == task]
                    tasks[source + ":" + task] = _metrics(
                        [values[i] for i in ids],
                        [rows[i]["options"].index(rows[i]["target"]) for i in ids],
                    )
                write(
                    f"predictions-{step}-{source}.json",
                    [
                        dict(
                            id=r["id"],
                            options=r["options"],
                            target=r["target"],
                            probabilities=v.softmax(-1).tolist(),
                        )
                        for r, v in zip(rows, values, strict=True)
                    ],
                )
        if frozen_hash(model) != before:
            raise ValueError("frozen weights changed")
        score = sum(m["nll"] for m in metrics.values()) / 2
        if score < best:
            save("best-delta.pt")
            best = score
            best_step = step
            best_values = all_values
        used = train[: step * 8]
        row = dict(
            step=step,
            selector=score,
            best_step=best_step,
            **metrics,
            tasks=tasks,
            rolling_train_ce=total_loss / step if step else None,
            last_gradient_norm=norm,
            consumed=len(used),
            unique_questions=len({r["id"] for r in used}),
            unique_parent_questions=len({r.get("parent_id", r["id"]) for r in used}),
            unique_states=len({r["group"] for r in used}),
            source_counts=dict(Counter(r["source"] for r in used)),
            task_counts=dict(Counter(r["source"] + ":" + r["task"] for r in used)),
            evaluation_seconds=time.perf_counter() - tick,
            elapsed_seconds=time.perf_counter() - start,
        )
        history.append(row)
        write("history.json", history)
        print(
            json.dumps({k: row[k] for k in ["step", "selector", "best_step", "elapsed_seconds"]}),
            flush=True,
        )

    evaluate(0)
    if not args.smoke:
        previous = json.loads(
            (root / "text-coverage-training-01/history.json").read_text(encoding="utf-8")
        )
        reference = next(r for r in previous if r["step"] == 1792)
        for source in ["original", "curriculum"]:
            for k in ["accuracy", "nll", "brier", "ece_15_bins"]:
                if abs(history[0][source][k] - reference[source][k]) > 1e-6:
                    raise ValueError("initial parity")
    for step in range(1, steps + 1):
        i = (step - 1) * 8
        scores, widths = forward(encoded_train[i : i + 8])
        target = torch.tensor(
            [r["options"].index(r["target"]) for r in train[i : i + 8]], device=device
        )
        loss = choice_loss(scores, widths, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("frozen gradient")
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
            raise ValueError("zero gradient")
        optimizer.step()
        total_loss += float(loss.detach())
        if step % 128 == 0 or step == steps:
            evaluate(step)
    save("final-delta.pt")
    model.load_state_dict(
        torch.load(args.output / "best-delta.pt", weights_only=True), strict=False
    )
    with torch.inference_mode():
        for encoded, expected in zip(encoded_val, best_values, strict=True):
            actual = []
            for i in range(0, len(encoded), 8):
                scores, widths = forward(encoded[i : i + 8])
                actual.extend(v.cpu() for v in scores.split(widths))
            if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise ValueError("reload mismatch")
    if sha(checkpoint) != "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5":
        raise ValueError("source checkpoint changed")
    write(
        "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            smoke=args.smoke,
            steps=steps,
            best_step=best_step,
            best_sha256=sha(args.output / "best-delta.pt"),
            final_sha256=sha(args.output / "final-delta.pt"),
            frozen_sha256=before,
            reload_exact=True,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            exact_resume=False,
            ended_utc=datetime.now(UTC).isoformat(),
        ),
    )


if __name__ == "__main__":
    main()
