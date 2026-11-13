"""Fixed two-update last-four-layer feasibility; no validation quality tuning."""

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.train_conditional_repair import verify_splits
from scripts.train_dinov2_image_probe import sha, write
from scripts.train_frozen_text_probe import _load_examples
from scripts.train_text_crossencoder import frozen_hash
from tiny_omni_decision.schema import DatasetManifest
from tiny_omni_decision.small_decoder_prompt import decision_prompt, option_labels

WEIGHTS = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
REFERENCE = "0ad927647362433e661c3805e61f981e488cf3f78da7c5de5a32c53f268c1282"


def load_data(root):
    config_path = root / "conditional-repair-comparison-vocabulary-01/config.json"
    if sha(config_path) != REFERENCE:
        raise ValueError("ID reference changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {Path(p): h for p, h in config["paths"].items() if p.endswith(".jsonl")}
    for p, h in paths.items():
        if sha(p) != h:
            raise ValueError("corpus changed")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )
    original = {}
    for split in ["train", "validation"]:
        kinds = {}
        for line in (root / f"text/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            for key, value in json.loads(raw["questions"]).items():
                kinds[f"{raw['state_id']}:{key}"] = value["type"]
        loaded = _load_examples(
            root / f"text/{split}.jsonl", manifest.model_copy(update={"split": split}), True
        )
        original[split] = [
            dict(
                id=e.id,
                state=e.state,
                question=e.question,
                options=e.options,
                target=e.target,
                source="original",
                task=kinds[e.source_record_id],
                group=e.source_record_id.split(":")[0],
            )
            for e in loaded
        ]
    corpus = {}
    for split, path in [
        ("train", root / "comparison-vocabulary-curriculum-01/train.jsonl"),
        ("validation", root / "conditional-curriculum-01/validation.jsonl"),
    ]:
        corpus[split] = original[split] + [
            dict(json.loads(s), source="curriculum")
            for s in path.read_text(encoding="utf-8").splitlines()
        ]
    verify_splits(corpus["train"], corpus["validation"])
    by_id = {r["id"]: r for rows in corpus.values() for r in rows}
    train = [by_id[i] for i in config["train_ids"]]
    validation = [[by_id[i] for i in ids] for ids in config["validation_ids"]]
    return train, validation, {str(p): h for p, h in paths.items()}


def encode_rows(tokenizer, rows):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": decision_prompt(r)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    return tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]


def configure(model):
    prefixes = tuple(f"model.layers.{i}." for i in range(26, 30)) + ("model.norm.",)
    names = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
        if p.requires_grad:
            names.append(name)
    if not names or len(model.model.layers) != 30:
        raise ValueError("unexpected architecture")
    return names


def batch_inputs(tokenizer, encoded, device):
    batch = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt").to(device)
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).masked_fill(
        batch["attention_mask"] == 0, 0
    )
    return batch


def candidate_logits(model, inputs, label_ids):
    hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
    return F.linear(hidden, model.lm_head.weight[label_ids])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "model", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.model / "model.safetensors") != WEIGHTS:
        raise ValueError("model changed")
    train, validation, paths = load_data(a.root)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_rows = train + sum(validation, [])
    max_options = max(len(r["options"]) for r in all_rows)
    encoded_labels = [
        tokenizer.encode(x, add_special_tokens=False) for x in option_labels(max_options)
    ]
    if (
        any(len(x) != 1 for x in encoded_labels)
        or len(set(x[0] for x in encoded_labels)) != max_options
    ):
        raise ValueError("labels not distinct single tokens")
    label_ids = [x[0] for x in encoded_labels]
    encoded = encode_rows(tokenizer, all_rows)
    lengths = [len(x) for x in encoded]
    if max(lengths) > 8192:
        raise ValueError("would exceed original context; no truncation allowed")
    a.output.mkdir(parents=True)
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        argv={k: str(v) for k, v in vars(a).items()},
        input_hashes=paths,
        reference_config_sha256=REFERENCE,
        label_ids=label_ids,
        train_ids=[r["id"] for r in train],
        validation_ids=[[r["id"] for r in v] for v in validation],
        prompt_lengths=dict(
            count=len(lengths),
            maximum=max(lengths),
            percentiles=np.quantile(lengths, [0.5, 0.9, 0.99]).tolist(),
        ),
        model_files={x.name: sha(x) for x in a.model.iterdir() if x.is_file()},
        versions={n: importlib.metadata.version(n) for n in ["torch", "transformers", "numpy"]},
        seed=17,
        optimizer="AdamW2e-5/wd.01/clip1",
        updates=2,
        smoke=True,
        trained_policy="last4decoder+finalnorm",
        validation_quality_evaluated=False,
    )
    write(a.output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    names = configure(model)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before = frozen_hash(model)
    label_tensor = torch.tensor(label_ids, device="cuda")
    first = batch_inputs(tokenizer, encoded[:8], "cuda")
    with torch.inference_mode():
        selected = candidate_logits(model, first, label_tensor)
        full = model(**first, use_cache=False, logits_to_keep=1).logits[:, -1, label_ids]
        delta = float((selected - full).abs().max())
        torch.testing.assert_close(selected, full, atol=1e-5, rtol=1e-5)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )
    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(2):
        tick = time.perf_counter()
        batch = encoded[step * 8 : (step + 1) * 8]
        rows = train[step * 8 : (step + 1) * 8]
        inputs = batch_inputs(tokenizer, batch, "cuda")
        logits = candidate_logits(model, inputs, label_tensor)
        losses = [
            F.cross_entropy(
                logits[i, : len(r["options"])][None],
                torch.tensor([r["options"].index(r["target"])], device="cuda"),
            )
            for i, r in enumerate(rows)
        ]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("loss")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("frozen gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True
        )
        if float(norm) <= 0:
            raise ValueError("no gradient")
        optimizer.step()
        torch.cuda.synchronize()
        history.append(
            dict(
                step=step + 1,
                train_ce=float(loss.detach()),
                gradient_norm=float(norm),
                seconds=time.perf_counter() - tick,
            )
        )
    training_peak = torch.cuda.max_memory_allocated()
    longest = sorted(range(len(train)), key=lambda i: len(encoded[i]), reverse=True)[:8]
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        long_scores = candidate_logits(
            model, batch_inputs(tokenizer, [encoded[i] for i in longest], "cuda"), label_tensor
        )
        if not torch.isfinite(long_scores).all():
            raise FloatingPointError("long batch")
    torch.cuda.synchronize()
    long_seconds = time.perf_counter() - tick
    long_peak = torch.cuda.max_memory_allocated()
    delta_state = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    torch.save(delta_state, a.output / "smoke-delta.pt")
    with torch.inference_mode():
        expected = candidate_logits(model, first, label_tensor).clone()
    saved = torch.load(a.output / "smoke-delta.pt", weights_only=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    with torch.inference_mode():
        actual = candidate_logits(model, first, label_tensor)
    assert torch.equal(expected, actual)
    after = frozen_hash(model)
    if before != after:
        raise ValueError("frozen parameters changed")
    write(
        a.output / "result.json",
        dict(
            status="PASS_FEASIBILITY_ONLY",
            history=history,
            parameters=parameters,
            trainable_parameters=trainable,
            trainable_names=names,
            frozen_sha256=before,
            projection_max_abs=delta,
            reload_exact=True,
            training_peak_bytes=training_peak,
            longest_batch_peak_bytes=long_peak,
            longest_batch_seconds=long_seconds,
            longest_batch_lengths=[len(encoded[i]) for i in longest],
            total_seconds=time.perf_counter() - start,
            delta_sha256=sha(a.output / "smoke-delta.pt"),
            delta_bytes=(a.output / "smoke-delta.pt").stat().st_size,
            gpu=torch.cuda.get_device_name(),
            cost_usd=0,
            validation_quality_evaluated=False,
            exact_resume=False,
        ),
    )
    print(
        json.dumps(
            dict(
                parameters=parameters,
                trainable=trainable,
                history=history,
                lengths=config["prompt_lengths"],
                training_peak=training_peak,
                long_seconds=long_seconds,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
