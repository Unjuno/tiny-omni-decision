"""Fixed 2048-update instruction-decoder learning curve; validation-only selection."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit_small_decoder_training import (
    WEIGHTS,
    batch_inputs,
    configure,
    encode_rows,
    load_data,
)
from scripts.train_dinov2_image_probe import sha, write
from scripts.train_frozen_text_probe import _metrics
from scripts.train_text_crossencoder import frozen_hash
from tiny_omni_decision.small_decoder_prompt import option_labels


def main():
    import subprocess

    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "model", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.model / "model.safetensors") != WEIGHTS:
        raise ValueError("base hash")
    train, validations, paths = load_data(a.root)
    assert len(train) == 16384 and [len(v) for v in validations] == [2540, 2048]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = train + sum(validations, [])
    tokens = encode_rows(tokenizer, rows)
    lengths = [len(x) for x in tokens]
    if max(lengths) > 1029:
        raise ValueError("preflight length changed")
    max_options = max(len(r["options"]) for r in rows)
    labels = [tokenizer.encode(s, add_special_tokens=False) for s in option_labels(max_options)]
    if any(len(x) != 1 for x in labels) or len(set(x[0] for x in labels)) != max_options:
        raise ValueError("label token contract")
    label_ids = [x[0] for x in labels]
    train_tokens = tokens[: len(train)]
    validation_tokens = [tokens[len(train) : len(train) + 2540], tokens[-2048:]]
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    names = configure(model)
    frozen_before = frozen_hash(model)
    config = dict(
        argv={k: str(v) for k, v in vars(a).items()},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        started_utc=datetime.now(UTC).isoformat(),
        seed=17,
        trainable_names=names,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        input_hashes=paths,
        base_sha256=WEIGHTS,
        label_ids=label_ids,
        max_tokens=max(lengths),
        train_ids=[r["id"] for r in train],
        validation_ids=[[r["id"] for r in v] for v in validations],
        max_steps=2048,
        eval_interval=256,
        lr=2e-5,
        weight_decay=0.01,
        clip=1,
        selector="earliest min mean of source NLL, including0",
        dropout=False,
        exact_resume=False,
    )
    write(a.output / "config.json", config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )
    label_tensor = torch.tensor(label_ids, device="cuda")
    history = []
    best = float("inf")
    best_step = 0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def scores(encoded, full=False):
        inputs = batch_inputs(tokenizer, encoded, "cuda")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
        if full:
            logits = model.lm_head(hidden)
            return logits[:, label_ids], logits.logsumexp(-1)
        return F.linear(hidden, model.lm_head.weight[label_tensor])

    def evaluate(step, save=True):
        report = []
        predictions = []
        tasks = {}
        tick = time.perf_counter()
        with torch.inference_mode():
            for source, (data, encoded) in enumerate(
                zip(validations, validation_tokens, strict=True)
            ):
                values = []
                mass = []
                for offset in range(0, len(data), 16):
                    logits, normalizer = scores(encoded[offset : offset + 16], True)
                    for i, row in enumerate(data[offset : offset + 16]):
                        choice = logits[i, : len(row["options"])]
                        if not torch.isfinite(choice).all():
                            raise FloatingPointError("validation")
                        values.append(choice.cpu())
                        mass.append(float((choice.logsumexp(-1) - normalizer[i]).exp()))
                targets = [r["options"].index(r["target"]) for r in data]
                metrics = _metrics(values, targets)
                metrics["mean_open_vocabulary_label_mass"] = float(np.mean(mass))
                report.append(metrics)
                for kind in sorted({r["task"] for r in data}):
                    ids = [i for i, r in enumerate(data) if r["task"] == kind]
                    tasks[f"{source}:{kind}"] = _metrics(
                        [values[i] for i in ids], [targets[i] for i in ids]
                    )
                pred = [
                    dict(
                        id=r["id"],
                        options=r["options"],
                        target=r["target"],
                        probabilities=v.softmax(-1).tolist(),
                        label_mass=m,
                    )
                    for r, v, m in zip(data, values, mass, strict=True)
                ]
                predictions.append(pred)
                if save:
                    write(a.output / f"step-{step:04d}-source-{source}.json", pred)
        return report, predictions, tasks, time.perf_counter() - tick

    losses = []
    norms = []
    for step in range(2049):
        if step:
            ids = range((step - 1) * 8, step * 8)
            batch = [train[i] for i in ids]
            logits = scores([train_tokens[i] for i in ids])
            loss = torch.stack(
                [
                    F.cross_entropy(
                        logits[i, : len(r["options"])][None],
                        torch.tensor([r["options"].index(r["target"])], device="cuda"),
                    )
                    for i, r in enumerate(batch)
                ]
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError("frozen gradient")
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True
            )
            if float(norm) <= 0:
                raise ValueError("no gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        if step % 256 == 0:
            metrics, predictions, tasks, seconds = evaluate(step)
            selector = sum(m["nll"] for m in metrics) / 2
            if selector < best:
                best = selector
                best_step = step
                best_predictions = predictions
                torch.save(
                    {
                        n: p.detach().cpu().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    },
                    a.output / "best-delta.pt",
                )
            consumed = train[: step * 8]
            record = dict(
                step=step,
                selector=selector,
                best_step=best_step,
                original=metrics[0],
                curriculum=metrics[1],
                tasks=tasks,
                rolling_train_ce=float(np.mean(losses)) if losses else None,
                mean_gradient_norm=float(np.mean(norms)) if norms else None,
                consumed=len(consumed),
                unique_questions=len({r["id"] for r in consumed}),
                unique_states=len({r["group"] for r in consumed}),
                source_counts=dict(Counter(r["source"] for r in consumed)),
                lr=2e-5,
                evaluation_seconds=seconds,
                elapsed_seconds=time.perf_counter() - start,
            )
            history.append(record)
            write(a.output / "history.json", history)
            losses = []
            norms = []
            print(
                json.dumps(
                    dict(
                        step=step,
                        selector=selector,
                        best_step=best_step,
                        elapsed_seconds=record["elapsed_seconds"],
                    )
                ),
                flush=True,
            )
    torch.save(
        {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
        a.output / "final-delta.pt",
    )
    saved = torch.load(a.output / "best-delta.pt", weights_only=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(saved[n])
    _, reloaded, _, _ = evaluate(best_step, save=False)
    if reloaded != best_predictions:
        raise ValueError("best reload probabilities differ")
    if frozen_hash(model) != frozen_before:
        raise ValueError("frozen tensors changed")
    write(
        a.output / "result.json",
        dict(
            status="PASS",
            steps=2048,
            best_step=best_step,
            frozen_sha256=frozen_before,
            best_sha256=sha(a.output / "best-delta.pt"),
            final_sha256=sha(a.output / "final-delta.pt"),
            reload_exact=True,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            parameters=config["parameters"],
            trainable_parameters=config["trainable_parameters"],
            gpu=torch.cuda.get_device_name(),
            cost_usd=0,
            exact_resume=False,
            finished_utc=datetime.now(UTC).isoformat(),
        ),
    )


if __name__ == "__main__":
    main()
