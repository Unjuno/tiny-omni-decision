"""Fixed all-position balancing diagnostic; inference only on observed development data."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit_small_decoder_training import (
    WEIGHTS,
    batch_inputs,
    candidate_logits,
    configure,
    encode_rows,
)
from scripts.evaluate_decoder_transfer import INPUT, require_selection
from scripts.range_generalization import sha, write
from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.candidate_rotation import align_mean, rotations

DELTA = "0caea48debd72e7ca4fa7c4f31da99005d8e6d587f07f32a3f1e9603cc3203e7"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "model", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.root / "small-decoder-learning-01"
    previous = args.root / "decoder-transfer-eval-01"
    result = read(run / "result.json")
    step = require_selection(result, read(run / "history.json"))
    inputs = args.root / "decoder-transfer-inputs-01/inputs.jsonl"
    if (
        sha(inputs) != INPUT
        or sha(args.model / "model.safetensors") != WEIGHTS
        or sha(run / "best-delta.pt") != DELTA
        or result["best_sha256"] != DELTA
    ):
        raise ValueError("pinned inputs changed")
    rows = [json.loads(s) for s in inputs.read_text().splitlines()]
    prior_selection = read(previous / "selection.json")
    if prior_selection["decoder_delta_sha256"] != DELTA:
        raise ValueError("comparison model differs")
    reference_hashes = read(Path("docs/pretrained_reuse/R2_SMALL_DECODER_RESULTS_20261012.json"))[
        "decoder-transfer-eval-01"
    ]["hashes"]
    for arm in ["base", "option_reverse"]:
        filename = f"decoder-{arm}-predictions.json"
        if sha(previous / filename) != reference_hashes[filename]:
            raise ValueError("previous predictions changed")
    args.output.mkdir(parents=True)
    write(
        args.output / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            argv={k: str(v) for k, v in vars(args).items()},
            runner_sha256=sha(Path(__file__)),
            base_sha256=WEIGHTS,
            delta_sha256=DELTA,
            input_sha256=INPUT,
            selected_step=step,
            aggregation="arithmetic_mean_identity_aligned_probabilities",
            temperature=1,
            training=False,
            reference_hashes=reference_hashes,
        ),
    )
    torch.set_num_threads(2)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    )
    expected = configure(model)
    weights = torch.load(run / "best-delta.pt", weights_only=True)
    if set(expected) != set(weights):
        raise ValueError("delta names differ")
    model.load_state_dict(weights, strict=False)
    model.eval().requires_grad_(False).cuda()
    labels = torch.tensor(read(run / "config.json")["label_ids"], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    all_predictions, metrics, comparisons = {}, {}, {}
    counts, max_difference = {}, 0.0
    for arm in ["base", "option_reverse"]:
        data = [r for r in rows if r["variant"] == arm]
        assert len(data) == 1024 and len({r["group"] for r in data}) == 128
        expanded = [dict(r, options=options) for r in data for options in rotations(r["options"])]
        encoded = encode_rows(tokenizer, expanded)
        if max(map(len, encoded)) > 222:
            raise ValueError("unexpected length; no truncation")
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(expanded), 16):
                logits = candidate_logits(
                    model, batch_inputs(tokenizer, encoded[offset : offset + 16], "cuda"), labels
                ).cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite")
                scores.extend(
                    logits[i, : len(r["options"])].softmax(-1).tolist()
                    for i, r in enumerate(expanded[offset : offset + 16])
                )
        prior = read(previous / f"decoder-{arm}-predictions.json")
        offset, predictions, values, targets, groups = 0, [], [], [], {}
        for row, old in zip(data, prior, strict=True):
            if any(row[k] != old[k] for k in ["id", "options", "target"]):
                raise ValueError("comparison identity differs")
            n = len(row["options"])
            rotated = scores[offset : offset + n]
            difference = float(np.max(np.abs(np.array(rotated[0]) - old["probabilities"])))
            max_difference = max(max_difference, difference)
            if difference > 1e-5:
                raise ValueError("rotation0 differs from saved predictions")
            probabilities = align_mean(row["options"], rotated)
            target = row["options"].index(row["target"])
            groups.setdefault(row["group"], []).append(
                int(np.argmax(probabilities) == target)
                - int(np.argmax(old["probabilities"]) == target)
            )
            values.append(torch.tensor(probabilities).log())
            targets.append(target)
            predictions.append(
                dict(
                    id=row["id"],
                    options=row["options"],
                    target=row["target"],
                    probabilities=probabilities,
                    rotation_probabilities=rotated,
                )
            )
            offset += n
        assert offset == len(scores)
        write(args.output / f"{arm}-predictions.json", predictions)
        all_predictions[arm] = predictions
        metrics[arm] = dict(overall=_metrics(values, targets), tasks={})
        for task in sorted({r["task"] for r in data}):
            indices = [i for i, r in enumerate(data) if r["task"] == task]
            metrics[arm]["tasks"][task] = _metrics(
                [values[i] for i in indices], [targets[i] for i in indices]
            )
        diffs = np.array([np.mean(v) for v in groups.values()])
        rng = np.random.default_rng(17)
        draws = diffs[rng.integers(len(diffs), size=(10000, len(diffs)))].mean(1)
        comparisons[arm] = dict(
            difference=float(diffs.mean()),
            percentile_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
        )
        counts[arm] = dict(questions=len(data), forward_examples=len(expanded))
        print(arm, metrics[arm]["overall"], flush=True)

    def choice(prediction):
        return prediction["options"][int(np.argmax(prediction["probabilities"]))]

    consistency = (
        sum(
            choice(a) == choice(b)
            for a, b in zip(all_predictions["base"], all_predictions["option_reverse"], strict=True)
        )
        / 1024
    )
    write(
        args.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            metrics=metrics,
            comparisons=comparisons,
            counts=counts,
            candidate_reversal_identity_consistency=consistency,
            rotation0_max_probability_difference=max_difference,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
            training=False,
            cost_usd=0,
            limitation="Observed development states; cyclic averaging lacks full permutation invariance.",
        ),
    )


if __name__ == "__main__":
    main()
