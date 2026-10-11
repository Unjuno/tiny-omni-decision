"""Fixed pooled-video interaction comparison with zero-video and scene-shuffle controls."""

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import _prediction_rows, _scene_derangement
from scripts.range_generalization import sha, write
from scripts.train_frozen_clevrer_binary_reasoning import (
    QUESTION_TYPES,
    _decision_metrics,
    _sample_weights,
    _validate_inputs,
)
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_video_probe import _combine, _predict, _text_features
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer


def make_scorer(policy):
    return (
        CandidateScorer(2304, 128)
        if policy == "concat"
        else BilinearCandidateScorer(768, 1536, 128)
    )


def paired_macro(rows, first, second):
    scenes = sorted({r["scene_index"] for r in rows})
    counts = np.zeros((len(scenes), 3))
    differences = np.zeros_like(counts)
    scene_index = {s: i for i, s in enumerate(scenes)}
    for row, a, b in zip(rows, first, second, strict=True):
        if a["id"] != b["id"] or row["id"] != a["id"]:
            raise ValueError("paired ID mismatch")
        i, j = scene_index[row["scene_index"]], QUESTION_TYPES.index(row["question_type"])
        counts[i, j] += 1
        differences[i, j] += int(b["prediction_index"] == b["target_index"]) - int(
            a["prediction_index"] == a["target_index"]
        )
    draws = np.random.default_rng(17).integers(len(scenes), size=(10000, len(scenes)))
    values = (differences[draws].sum(1) / counts[draws].sum(1)).mean(1)
    return dict(
        difference=float((differences.sum(0) / counts.sum(0)).mean()),
        percentile_95_interval=np.quantile(values, [0.025, 0.975]).tolist(),
        scenes=len(scenes),
        replicates=10000,
        seed=17,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    checked = _validate_inputs(args.config, config)
    rows = {"train": checked["train_rows"], "validation": checked["validation_rows"]}
    if [len(rows[s]) for s in rows] != [3794, 3014]:
        raise ValueError("fixed corpus count differs")
    if args.smoke:
        rows = {
            split: sum(
                (
                    [r for r in data if r["question_type"] == kind][:2]
                    + [r for r in data if r["question_type"] == kind][-2:]
                    for kind in QUESTION_TYPES
                ),
                [],
            )
            for split, data in rows.items()
        }
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_encoder"]["model_path"], local_files_only=True
    )
    encoder = (
        AutoModel.from_pretrained(config["text_encoder"]["model_path"], local_files_only=True)
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    text, targets, videos, truncation = {}, {}, {}, {}
    for split, examples in rows.items():
        lengths = tokenizer([f"Question: {r['question']}" for r in examples], truncation=False)[
            "input_ids"
        ]
        truncation[split] = dict(
            maximum=max(map(len, lengths)), over128=sum(len(x) > 128 for x in lengths)
        )
        text[split], targets[split] = _text_features(
            examples, tokenizer, encoder, torch.device("cuda")
        )
        videos[split] = torch.from_numpy(
            np.stack([checked["feature_by_hash"][r["media_sha256"]] for r in examples]).copy()
        )
    del encoder, tokenizer
    torch.cuda.empty_cache()
    initial_rng = torch.get_rng_state()
    epochs = 1 if args.smoke else 24
    write(
        args.output / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            argv={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            runner_sha256=sha(Path(__file__)),
            source_config=config,
            source_config_sha256=checked["config_sha256"],
            feature_cache_sha256=checked["feature_cache_sha256"],
            sample_manifest_sha256=checked["sample_manifest_sha256"],
            source_question_sha256=checked["source_question_sha256"],
            seed=17,
            epochs=epochs,
            truncation=truncation,
            torch=torch.__version__,
            precision="float32",
            gpu=torch.cuda.get_device_name(),
            selector="earliest minimum validation macro_question_type NLL",
            ids={s: [r["id"] for r in v] for s, v in rows.items()},
        ),
    )
    weights = _sample_weights(rows["train"])
    results, predictions = {}, {}
    order_reference = None
    mapping = _scene_derangement([r["scene_index"] for r in rows["validation"]], 17)
    by_scene = {
        r["scene_index"]: v for r, v in zip(rows["validation"], videos["validation"], strict=True)
    }
    shuffled_video = torch.stack([by_scene[mapping[r["scene_index"]]] for r in rows["validation"]])
    for policy in ["concat", "bilinear", "no_video"]:
        out = args.output / policy
        out.mkdir()
        features = {
            s: _combine(torch.zeros_like(v) if policy == "no_video" else v, text[s])
            for s, v in videos.items()
        }
        torch.set_rng_state(initial_rng)
        model = make_scorer(policy).cuda()
        initial_digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            initial_digest.update(name.encode())
            initial_digest.update(value.detach().cpu().numpy().tobytes())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        generator = torch.Generator().manual_seed(17)
        digest = hashlib.sha256()
        history, best_logits = [], None
        best, best_epoch = float("inf"), 0
        tick = time.perf_counter()
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(rows["train"]), generator=generator).tolist()
            digest.update(np.asarray(order, dtype="<i8").tobytes())
            losses = []
            model.train()
            for offset in range(0, len(order), 64):
                ids = order[offset : offset + 64]
                logits = model(torch.cat([features["train"][i] for i in ids]).cuda()).reshape(-1, 2)
                loss = (
                    (
                        F.cross_entropy(
                            logits,
                            torch.tensor([targets["train"][i] for i in ids], device="cuda"),
                            reduction="none",
                        )
                        * weights[ids].cuda()
                    ).sum()
                    * len(order)
                    / len(ids)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(loss) or any(
                    not torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise ValueError("nonfinite loss or gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
            values = _predict(model, features["validation"], torch.device("cuda"))
            metric = _decision_metrics(values, targets["validation"], rows["validation"])
            train_metric = _decision_metrics(
                _predict(model, features["train"], torch.device("cuda")),
                targets["train"],
                rows["train"],
            )
            history.append(
                dict(
                    epoch=epoch,
                    rolling_loss=float(np.mean(losses)),
                    train=train_metric,
                    validation=metric,
                )
            )
            write(out / "history.json", history)
            score = metric["macro_question_type"]["nll"]
            if score < best:
                best, best_epoch = score, epoch
                best_logits = [x.clone() for x in values]
                torch.save(model.state_dict(), out / "best.pt")
            print(policy, epoch, metric["macro_question_type"], flush=True)
        torch.save(model.state_dict(), out / "final.pt")
        model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        reloaded = _predict(model, features["validation"], torch.device("cuda"))
        if not all(torch.equal(a, b) for a, b in zip(reloaded, best_logits, strict=True)):
            raise ValueError("reload differs")
        shuffled = _predict(
            model,
            _combine(
                torch.zeros_like(shuffled_video) if policy == "no_video" else shuffled_video,
                text["validation"],
            ),
            torch.device("cuda"),
        )
        predictions[policy] = _prediction_rows(
            rows["validation"], reloaded, targets["validation"], policy
        )
        predictions[policy + "_shuffled"] = _prediction_rows(
            rows["validation"], shuffled, targets["validation"], policy
        )
        write(out / "predictions.json", predictions[policy])
        write(out / "shuffled-predictions.json", predictions[policy + "_shuffled"])
        order_hash = digest.hexdigest()
        if order_reference is not None and order_hash != order_reference:
            raise ValueError("sample order differs")
        order_reference = order_hash
        results[policy] = dict(
            initialization_sha256=initial_digest.hexdigest(),
            best_epoch=best_epoch,
            reload_exact=True,
            parameters=sum(p.numel() for p in model.parameters()),
            matched=_decision_metrics(reloaded, targets["validation"], rows["validation"]),
            shuffled=_decision_metrics(shuffled, targets["validation"], rows["validation"]),
            consumed=epochs * len(rows["train"]),
            selected_consumed=best_epoch * len(rows["train"]),
            unique_examples=len(rows["train"]),
            scenes=len({r["scene_index"] for r in rows["train"]}),
            best_sha256=sha(out / "best.pt"),
            final_sha256=sha(out / "final.pt"),
            seconds=time.perf_counter() - tick,
            order_sha256=order_hash,
        )
        write(out / "result.json", results[policy])
        del model, optimizer, features
        torch.cuda.empty_cache()
    if results["bilinear"]["initialization_sha256"] != results["no_video"]["initialization_sha256"]:
        raise ValueError("bilinear controls initialized differently")
    if (
        sha(Path(config["text_encoder"]["model_path"]) / "model.safetensors")
        != checked["text_model_weights_sha256"]
        or sha(Path(config["features"]["cache"])) != checked["feature_cache_sha256"]
    ):
        raise ValueError("frozen inputs changed")
    comparisons = {
        name: paired_macro(rows["validation"], predictions[a], predictions[b])
        for name, a, b in [
            ("bilinear_minus_concat", "concat", "bilinear"),
            ("bilinear_minus_no_video", "no_video", "bilinear"),
            ("matched_minus_shuffled", "bilinear_shuffled", "bilinear"),
        ]
    }
    write(
        args.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            smoke=args.smoke,
            conditions=results,
            comparisons=comparisons,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            cost_usd=0,
            limitation="Previously observed selection validation; not blind video quality.",
        ),
    )


if __name__ == "__main__":
    main()
