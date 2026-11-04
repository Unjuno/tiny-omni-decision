"""One fixed post-selection transfer comparison; never select from these scores."""

import argparse
import json
import math
import time
from pathlib import Path

from scripts.prepare_decoder_transfer import VARIANTS
from scripts.range_generalization import sha, write

INPUT = "dc5871692646473f8eac216aa7e0e9ee065040a4c5c9f6e3e22c0a16099ef51d"
BERT_DELTA = "868cc435f3b70ce84d0c49781d80680ba28a1078cbcf6de25f53a58756661b13"


def require_selection(result, history):
    if (
        result.get("status") != "PASS"
        or result.get("steps") != 2048
        or not result.get("reload_exact")
    ):
        raise ValueError("complete run/reload required")
    if [x["step"] for x in history] != list(range(0, 2049, 256)):
        raise ValueError("incomplete learning curve")
    if any(not math.isfinite(x["selector"]) for x in history):
        raise ValueError("nonfinite selector")
    best = min(history, key=lambda x: x["selector"])["step"]
    if best != result["best_step"]:
        raise ValueError("wrong selection")
    return best


def main():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    from scripts.audit_small_decoder_training import WEIGHTS, batch_inputs, configure, encode_rows
    from scripts.evaluate_vocabulary_transfer import opposing_pairs
    from scripts.train_conditional_repair import configure_repair_policy
    from scripts.train_frozen_text_probe import _metrics
    from tiny_omni_decision.small_decoder_prompt import option_labels

    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "model", "bert", "data", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    run = a.root / "small-decoder-learning-01"
    result, history, config = (
        read(run / f"{name}.json") for name in ["result", "history", "config"]
    )
    selected = require_selection(result, history)
    if (
        config["seed"] != 17
        or config["max_steps"] != 2048
        or config["trainable_parameters"] != 14160960
    ):
        raise ValueError("wrong experiment")
    for path, digest in config["input_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("training input changed")
    delta = run / "best-delta.pt"
    bert_delta = a.root / "conditional-repair-comparison-vocabulary-01/best-delta.pt"
    if sha(delta) != result["best_sha256"] or sha(bert_delta) != BERT_DELTA:
        raise ValueError("delta changed")
    if sha(a.model / "model.safetensors") != WEIGHTS:
        raise ValueError("base changed")
    if (
        sha(a.bert / "model.safetensors")
        != "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("BERT changed")
    if sha(a.data / "inputs.jsonl") != INPUT:
        raise ValueError("transfer input changed")
    rows = [
        json.loads(s) for s in (a.data / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6144 and len({r["id"] for r in rows}) == 6144
    a.output.mkdir(parents=True)
    write(
        a.output / "selection.json",
        dict(
            step=selected,
            result_sha256=sha(run / "result.json"),
            config_sha256=sha(run / "config.json"),
            history_sha256=sha(run / "history.json"),
            input_sha256=INPUT,
            decoder_delta_sha256=sha(delta),
            bert_delta_sha256=BERT_DELTA,
            runner_sha256=sha(Path(__file__)),
            no_selection_from_transfer=True,
        ),
    )
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    results = {}
    predictions = {}
    pairs = {}
    for name in ["base", "decoder", "bert"]:
        if name == "bert":
            tokenizer = AutoTokenizer.from_pretrained(a.bert, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                a.bert, local_files_only=True, dtype=torch.float32
            )
            expected = configure_repair_policy(model, "all_encoder_blocks")
            state = torch.load(bert_delta, weights_only=True)
            if set(state) != set(expected):
                raise ValueError("BERT target mismatch")
            model.load_state_dict(state, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                a.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
            )
            if name == "decoder":
                expected = configure(model)
                state = torch.load(delta, weights_only=True)
                if set(state) != set(expected):
                    raise ValueError("decoder target mismatch")
                model.load_state_dict(state, strict=False)
            labels = [
                tokenizer.encode(s, add_special_tokens=False)
                for s in option_labels(max(len(r["options"]) for r in rows))
            ]
            if any(len(x) != 1 for x in labels):
                raise ValueError("multi-token label")
            label_ids = torch.tensor([x[0] for x in labels], device="cuda")
        model.eval().requires_grad_(False).cuda()
        results[name] = {}
        predictions[name] = {}
        pairs[name] = {}
        with torch.inference_mode():
            for variant in VARIANTS:
                data = [r for r in rows if r["variant"] == variant]
                values = []
                max_tokens = 0
                encoded = encode_rows(tokenizer, data) if name != "bert" else None
                for offset in range(0, len(data), 16):
                    batch = data[offset : offset + 16]
                    if name == "bert":
                        first = [
                            f"Question: {r['question']}\nCandidate answer: {o}"
                            for r in batch
                            for o in r["options"]
                        ]
                        second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
                        inputs = tokenizer(
                            first, second, padding=True, truncation=False, return_tensors="pt"
                        )
                        max_tokens = max(max_tokens, inputs["input_ids"].shape[1])
                        if max_tokens > 512:
                            raise ValueError("BERT truncation")
                        logits = model(**inputs.to("cuda")).logits[:, 0].cpu()
                        values.extend(logits.split([len(r["options"]) for r in batch]))
                    else:
                        token_batch = encoded[offset : offset + 16]
                        max_tokens = max(max_tokens, max(map(len, token_batch)))
                        if max_tokens > 8192:
                            raise ValueError("decoder truncation")
                        hidden = model.model(
                            **batch_inputs(tokenizer, token_batch, "cuda"), use_cache=False
                        ).last_hidden_state[:, -1]
                        logits = F.linear(hidden, model.lm_head.weight[label_ids]).cpu()
                        values.extend(logits[i, : len(r["options"])] for i, r in enumerate(batch))
                if any(not torch.isfinite(v).all() for v in values):
                    raise ValueError("nonfinite")
                targets = [r["options"].index(r["target"]) for r in data]
                metrics = dict(overall=_metrics(values, targets), tasks={}, max_tokens=max_tokens)
                for task in sorted({r["task"] for r in data}):
                    ids = [i for i, r in enumerate(data) if r["task"] == task]
                    metrics["tasks"][task] = _metrics(
                        [values[i] for i in ids], [targets[i] for i in ids]
                    )
                pred = [
                    dict(
                        id=r["id"],
                        options=r["options"],
                        target=r["target"],
                        probabilities=v.softmax(-1).tolist(),
                    )
                    for r, v in zip(data, values, strict=True)
                ]
                write(a.output / f"{name}-{variant}-predictions.json", pred)
                results[name][variant] = metrics
                predictions[name][variant] = pred
                pairs[name][variant] = opposing_pairs(data, pred)
                print(name, variant, metrics["overall"]["accuracy"], flush=True)
        del model
        torch.cuda.empty_cache()
    comparisons = {}
    consistency = {}

    def correct(p):
        return (
            p["options"][max(range(len(p["options"])), key=lambda i: p["probabilities"][i])]
            == p["target"]
        )

    def chosen(p):
        return p["options"][max(range(len(p["options"])), key=lambda i: p["probabilities"][i])]

    for name in results:
        consistency[name] = {
            v: sum(
                chosen(x) == chosen(y)
                for x, y in zip(predictions[name]["base"], predictions[name][v], strict=True)
            )
            / 1024
            for v in ["option_reverse", "state_reverse"]
        }
        if name == "base":
            continue
        comparisons[name] = {}
        for variant in VARIANTS:
            data = [r for r in rows if r["variant"] == variant]
            groups = {}
            for row, x, y in zip(
                data, predictions["base"][variant], predictions[name][variant], strict=True
            ):
                groups.setdefault(row["group"], []).append(int(correct(y)) - int(correct(x)))
            diffs = np.array([np.mean(v) for v in groups.values()])
            rng = np.random.default_rng(17)
            draws = diffs[rng.integers(len(diffs), size=(10000, len(diffs)))].mean(1)
            comparisons[name][variant] = dict(
                difference=float(diffs.mean()),
                percentile_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
            )
    write(
        a.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            metrics=results,
            comparisons_to_base=comparisons,
            opposing_pairs=pairs,
            identity_prediction_consistency=consistency,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
            cost_usd=0,
            limitation="Known-template development,128states; no blind or equal-compute claim.",
        ),
    )


if __name__ == "__main__":
    main()
