"""Read-only selected-checkpoint fit on a predetermined consumed training slice."""

import argparse
import json
import time
from pathlib import Path

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
from scripts.evaluate_decoder_transfer import require_selection
from scripts.train_dinov2_image_probe import sha, write
from scripts.train_frozen_text_probe import _metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "model", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    run = a.root / "small-decoder-learning-01"
    result = json.loads((run / "result.json").read_text())
    history = json.loads((run / "history.json").read_text())
    step = require_selection(result, history)
    if step * 8 < 1024:
        raise ValueError("selected checkpoint has not consumed fixed slice")
    if (
        sha(a.model / "model.safetensors") != WEIGHTS
        or sha(run / "best-delta.pt") != result["best_sha256"]
    ):
        raise ValueError("weights changed")
    train, _, paths = load_data(a.root)
    rows = train[:1024]
    a.output.mkdir(parents=True)
    write(a.output / "inputs.json", rows)
    write(
        a.output / "selection.json",
        dict(
            step=step,
            delta_sha256=result["best_sha256"],
            input_sha256=sha(a.output / "inputs.json"),
            source_hashes=paths,
            runner_sha256=sha(Path(__file__)),
            selection_from_training_fit=False,
        ),
    )
    torch.set_num_threads(2)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = json.loads((run / "config.json").read_text())
    ids = torch.tensor(config["label_ids"], device="cuda")
    encoded = encode_rows(tokenizer, rows)
    if max(map(len, encoded)) > 1029:
        raise ValueError("unexpected length; no truncation")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    )
    names = configure(model)
    state = torch.load(run / "best-delta.pt", weights_only=True)
    if set(names) != set(state):
        raise ValueError("delta names differ")
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False).cuda()
    torch.cuda.reset_peak_memory_stats()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            hidden = model.model(
                **batch_inputs(tokenizer, encoded[offset : offset + 16], "cuda"), use_cache=False
            ).last_hidden_state[:, -1]
            logits = F.linear(hidden, model.lm_head.weight[ids]).cpu()
            values.extend(
                logits[i, : len(r["options"])] for i, r in enumerate(rows[offset : offset + 16])
            )
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError("nonfinite")
    targets = [r["options"].index(r["target"]) for r in rows]
    metrics = {}
    for source in ["original", "curriculum"]:
        selected = [i for i, r in enumerate(rows) if r["source"] == source]
        assert len(selected) == 512
        metrics[source] = _metrics([values[i] for i in selected], [targets[i] for i in selected])
        for task in sorted({rows[i]["task"] for i in selected}):
            indexes = [i for i in selected if rows[i]["task"] == task]
            metrics[source + ":" + task] = _metrics(
                [values[i] for i in indexes], [targets[i] for i in indexes]
            )
    write(
        a.output / "predictions.json",
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
    write(
        a.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            metrics=metrics,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            training_performed=False,
            limitation="Already-seen training slice; not a generalization estimate.",
        ),
    )


if __name__ == "__main__":
    main()
