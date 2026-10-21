"""Shared versus independent text tower with matched video/no-video controls."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import _prediction_rows, _scene_derangement
from scripts.range_generalization import sha, write
from scripts.train_dinov2_spatial_control import tensor_hash
from scripts.train_frozen_clevrer_binary_reasoning import (
    QUESTION_TYPES,
    _decision_metrics,
    _sample_weights,
    _validate_inputs,
)
from scripts.train_frozen_video_probe import _text_features
from scripts.train_video_interaction_control import paired_macro
from tiny_omni_decision.token_readout import VideoTokenReadout
from tiny_omni_decision.token_transport import read_tokens

CROSS_MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
CROSS_WEIGHTS = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
CROSS_CHECKPOINT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/numeric-deterministic-control-01/best-delta.pt"
)
CROSS_DELTA = "0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882"
TOKEN_REFERENCE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/video-token-readout-01/extraction.json"
)
TOKEN_REFERENCE_HASH = "9855b80d0eea9d6ca82185c7735aed7bb911f18b8f60a4539008816ca6264e06"


def worker_environment(repo):
    env = dict(os.environ)
    parts = [str(repo / "src"), str(repo)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def text_token_contract(rows, tokenizer):
    texts = [f"Question: {r['question']}" for r in rows]
    texts += [f"Candidate answer: {o}" for r in rows for o in r["options"]]
    ids = tokenizer(texts, max_length=128, truncation=True)["input_ids"]
    return dict(
        strings=len(texts),
        at_token_cap=sum(len(v) == 128 for v in ids),
        ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "output", "audit", "worker-python"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    if (
        sha(CROSS_MODEL / "model.safetensors") != CROSS_WEIGHTS
        or sha(CROSS_CHECKPOINT) != CROSS_DELTA
        or sha(TOKEN_REFERENCE) != TOKEN_REFERENCE_HASH
    ):
        raise ValueError("frozen shared-text or token reference changed")
    expected_tokens = {
        r["media_sha256"]: r["token_sha256"] for r in json.loads(TOKEN_REFERENCE.read_text())
    }
    if shutil.disk_usage(args.output.parent).free < 64 * 1024**2 + 512 * 1024**2:
        raise OSError("need64MiB output allowance plus512MiB reserve")
    checked = _validate_inputs(args.config, config)
    rows = {s: checked[s + "_rows"] for s in ["train", "validation"]}
    if [len(rows[s]) for s in rows] != [3794, 3014]:
        raise ValueError("fixed corpus counts changed")
    if args.smoke:
        rows = {
            s: sum(
                (
                    [r for r in data if r["question_type"] == q][:2]
                    + [r for r in data if r["question_type"] == q][-2:]
                    for q in QUESTION_TYPES
                ),
                [],
            )
            for s, data in rows.items()
        }
    keys = list(dict.fromkeys(r["media_sha256"] for data in rows.values() for r in data))
    import psutil

    bank_bytes = len(keys) * 2048 * 768 * 4
    if psutil.virtual_memory().available < bank_bytes + 8 * 1024**3:
        raise ValueError("RAM reserve unavailable")
    args.output.mkdir(parents=True)
    write(args.output / "request.json", keys)
    epochs = 1 if args.smoke else 24
    write(
        args.output / "config.json",
        dict(
            argv=sys.argv,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            script_sha256=sha(Path(__file__)),
            source_config=config,
            source_config_sha256=checked["config_sha256"],
            sample_manifest_sha256=checked["sample_manifest_sha256"],
            source_question_sha256=checked["source_question_sha256"],
            feature_cache_sha256=checked["feature_cache_sha256"],
            seed=17,
            epochs=epochs,
            effective_batch=64,
            microbatch=8,
            selector="earliest minimum validation macro_question_type NLL",
            ids={s: [r["id"] for r in data] for s, data in rows.items()},
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            bank_bytes=bank_bytes,
            worker_pythonpath_prefix=[
                str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1]),
            ],
            cross_model=str(CROSS_MODEL),
            cross_weights_sha256=CROSS_WEIGHTS,
            cross_checkpoint=str(CROSS_CHECKPOINT),
            cross_checkpoint_sha256=CROSS_DELTA,
            token_reference_sha256=TOKEN_REFERENCE_HASH,
            conditions=[
                "minilm_attention",
                "minilm_no_video",
                "shared_attention",
                "shared_no_video",
            ],
        ),
    )
    start = time.perf_counter()
    bank = np.empty((len(keys), 2048, 768), dtype=np.float32)
    command = [
        str(args.worker_python),
        "-m",
        "scripts.audit_token_ram_transport",
        "--audit",
        str(args.audit),
        "--request",
        str(args.output / "request.json"),
        "--worker",
        "--live",
    ]
    token_records = []
    with (args.output / "worker.log").open("wb") as log:
        repo = Path(__file__).resolve().parents[1]
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=log, cwd=repo, env=worker_environment(repo)
        )
        try:
            for i, key in enumerate(keys):
                record = read_tokens(child.stdout, (8, 16, 16, 768))
                if record is None or record[0] != str(i):
                    raise ValueError("worker order mismatch")
                bank[i] = record[1].reshape(2048, 768)
                delta = float(np.abs(bank[i].mean(0) - checked["feature_by_hash"][key]).max())
                if delta > 1e-5:
                    raise ValueError(f"pooled parity failed: {delta}")
                token_digest = hashlib.sha256(bank[i].tobytes()).hexdigest()
                if token_digest != expected_tokens.get(key):
                    raise ValueError(f"token identity changed: {key}")
                token_records.append(
                    dict(
                        media_sha256=key,
                        pooled_max_abs=delta,
                        token_sha256=token_digest,
                    )
                )
                write(args.output / "extraction.json", token_records)
                if i % 32 == 0:
                    print("extracted", i + 1, "/", len(keys), flush=True)
            if read_tokens(child.stdout, (8, 16, 16, 768)) is not None or child.wait(30) != 0:
                raise ValueError("worker failed")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
    extraction_seconds = time.perf_counter() - start
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_encoder"]["model_path"], local_files_only=True
    )
    encoder = (
        AutoModel.from_pretrained(config["text_encoder"]["model_path"], local_files_only=True)
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    text, targets = {}, {}
    contracts = {s: text_token_contract(data, tokenizer) for s, data in rows.items()}
    text_parameter_counts = {"minilm": sum(p.numel() for p in encoder.parameters())}
    for s, data in rows.items():
        features, labels = _text_features(data, tokenizer, encoder, torch.device("cuda"))
        text[s], targets[s] = torch.stack(features).cuda(), torch.tensor(labels, device="cuda")
    del encoder, tokenizer
    initial_rng = torch.get_rng_state()
    text_banks = {"minilm": text}
    tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL, local_files_only=True)
    if contracts != {s: text_token_contract(data, tokenizer) for s, data in rows.items()}:
        raise ValueError("text tokenization differs between encoders")
    encoder = AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL, local_files_only=True)
    delta = torch.load(CROSS_CHECKPOINT, map_location="cpu", weights_only=True)
    expected = {
        n
        for n, _ in encoder.named_parameters()
        if n.startswith(("bert.encoder.layer.", "classifier."))
    }
    if set(delta) != expected:
        raise ValueError("shared checkpoint scope changed")
    encoder.load_state_dict(delta, strict=False)
    encoder.eval().requires_grad_(False).cuda()
    before = tensor_hash(encoder)
    text_parameter_counts["shared"] = sum(p.numel() for p in encoder.parameters())
    text_banks["shared"] = {}
    for split, data in rows.items():
        values, labels = _text_features(data, tokenizer, encoder.base_model, torch.device("cuda"))
        if labels != targets[split].tolist():
            raise ValueError("target order changed")
        text_banks["shared"][split] = torch.stack(values).cuda()
    if before != tensor_hash(encoder):
        raise ValueError("shared encoder mutated")
    write(
        args.output / "text-feature-contract.json",
        dict(
            contracts=contracts,
            parameters=text_parameter_counts,
            shared_frozen_sha256=before,
            feature_hashes={
                policy: {
                    split: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()
                    for split, v in bank.items()
                }
                for policy, bank in text_banks.items()
            },
        ),
    )
    del encoder, tokenizer, delta
    video = torch.from_numpy(bank).cuda()
    del bank
    indices = {
        s: torch.tensor([keys.index(r["media_sha256"]) for r in data], device="cuda")
        for s, data in rows.items()
    }
    mapping = _scene_derangement([r["scene_index"] for r in rows["validation"]], 17)
    by_scene = {r["scene_index"]: keys.index(r["media_sha256"]) for r in rows["validation"]}
    shuffled = torch.tensor(
        [by_scene[mapping[r["scene_index"]]] for r in rows["validation"]], device="cuda"
    )
    weights = _sample_weights(rows["train"]).cuda()
    results, predictions = {}, {}

    def predict(model, split, asset_indices=None):
        model.eval()
        lookup = indices[split] if asset_indices is None else asset_indices
        with torch.no_grad():
            return list(
                torch.cat(
                    [
                        model(video[lookup[i : i + 8]], text[split][i : i + 8]).cpu()
                        for i in range(0, len(rows[split]), 8)
                    ]
                )
            )

    for policy, mode in [(p, m) for p in ["minilm", "shared"] for m in ["attention", "no_video"]]:
        condition = policy + "_" + mode
        text = text_banks[policy]
        out = args.output / condition
        out.mkdir()
        torch.set_rng_state(initial_rng)
        model = VideoTokenReadout(mode).cuda()
        initial_hash = tensor_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        generator = torch.Generator().manual_seed(17)
        history, best_logits = [], None
        best, best_epoch = float("inf"), 0
        order_digest = hashlib.sha256()
        tick = time.perf_counter()
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(rows["train"]), generator=generator).tolist()
            order_digest.update(np.asarray(order, dtype="<i8").tobytes())
            model.train()
            for start_index in range(0, len(order), 64):
                batch = order[start_index : start_index + 64]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(batch), 8):
                    ids = batch[offset : offset + 8]
                    logits = model(video[indices["train"][ids]], text["train"][ids])
                    loss = (
                        (
                            F.cross_entropy(logits, targets["train"][ids], reduction="none")
                            * weights[ids]
                        ).sum()
                        * len(order)
                        / len(batch)
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite loss")
                    loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
            values = predict(model, "validation")
            metric = _decision_metrics(values, targets["validation"].tolist(), rows["validation"])
            train_metric = _decision_metrics(
                predict(model, "train"), targets["train"].tolist(), rows["train"]
            )
            history.append(dict(epoch=epoch, train=train_metric, validation=metric))
            write(out / "history.json", history)
            score = metric["macro_question_type"]["nll"]
            if score < best:
                best, best_epoch, best_logits = score, epoch, values
                torch.save(model.state_dict(), out / "best.pt")
            print(condition, epoch, metric["macro_question_type"], flush=True)
        torch.save(model.state_dict(), out / "final.pt")
        model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
        reloaded = predict(model, "validation")
        if not all(torch.equal(a, b) for a, b in zip(reloaded, best_logits, strict=True)):
            raise ValueError("best reload mismatch")
        for suffix, values in [
            ("", reloaded),
            ("_shuffled", predict(model, "validation", shuffled)),
        ]:
            predictions[condition + suffix] = _prediction_rows(
                rows["validation"], values, targets["validation"].tolist(), condition + suffix
            )
            write(out / ("predictions" + suffix + ".json"), predictions[condition + suffix])
        results[condition] = dict(
            text_policy=policy,
            text_parameters=text_parameter_counts[policy],
            initial_tensor_sha256=initial_hash,
            best_epoch=best_epoch,
            reload_exact=True,
            parameters=sum(p.numel() for p in model.parameters()),
            active_parameters=sum(
                p.numel()
                for n, p in model.named_parameters()
                if mode == "attention" or not n.startswith(("key.", "position."))
            ),
            matched=_decision_metrics(reloaded, targets["validation"].tolist(), rows["validation"]),
            consumed=epochs * len(rows["train"]),
            selected_consumed=best_epoch * len(rows["train"]),
            unique_examples=len(rows["train"]),
            unique_assets=len(set(indices["train"].tolist())),
            seconds=time.perf_counter() - tick,
            order_sha256=order_digest.hexdigest(),
            best_sha256=sha(out / "best.pt"),
            final_sha256=sha(out / "final.pt"),
        )
        write(out / "result.json", results[condition])
        del model, optimizer
    if len({r["initial_tensor_sha256"] for r in results.values()}) != 1:
        raise ValueError("head initialization differs")
    if len({r["order_sha256"] for r in results.values()}) != 1:
        raise ValueError("sample order differs")
    if (
        sha(Path(config["text_encoder"]["model_path"]) / "model.safetensors")
        != checked["text_model_weights_sha256"]
        or sha(Path(config["features"]["cache"])) != checked["feature_cache_sha256"]
    ):
        raise ValueError("frozen input changed during run")
    if (
        sha(CROSS_MODEL / "model.safetensors") != CROSS_WEIGHTS
        or sha(CROSS_CHECKPOINT) != CROSS_DELTA
    ):
        raise ValueError("shared frozen sources changed")
    comparisons = (
        {}
        if args.smoke
        else {
            name: paired_macro(rows["validation"], predictions[a], predictions[b])
            for name, a, b in [
                ("shared_minus_minilm_video", "minilm_attention", "shared_attention"),
                ("shared_minus_minilm_no_video", "minilm_no_video", "shared_no_video"),
                ("minilm_video_minus_no_video", "minilm_no_video", "minilm_attention"),
                ("shared_video_minus_no_video", "shared_no_video", "shared_attention"),
                ("minilm_matched_minus_shuffled", "minilm_attention_shuffled", "minilm_attention"),
                ("shared_matched_minus_shuffled", "shared_attention_shuffled", "shared_attention"),
            ]
        }
    )
    write(
        args.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            smoke=args.smoke,
            conditions=results,
            comparisons=comparisons,
            extraction_seconds=extraction_seconds,
            total_seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            cost_usd=0,
            limitation="Observed development validation, not blind/general video quality.",
        ),
    )


if __name__ == "__main__":
    main()
