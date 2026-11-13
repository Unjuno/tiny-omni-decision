"""Fixed 64-training-example fit diagnostic; no validation inference or selection."""

import argparse
import hashlib
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_clevrer_binary_reasoning_controls import _prediction_rows
from scripts.range_generalization import sha, write
from scripts.train_frozen_clevrer_binary_reasoning import (
    _decision_metrics,
    _sample_weights,
    _validate_inputs,
)
from scripts.train_frozen_video_probe import _text_features
from tiny_omni_decision.token_readout import VideoTokenReadout
from tiny_omni_decision.token_transport import read_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "output", "audit", "worker-python"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    checked = _validate_inputs(args.config, config)
    rows = checked["train_rows"][:64]
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64:
        raise ValueError("need 64 unique training choices")
    keys = list(dict.fromkeys(r["media_sha256"] for r in rows))
    args.output.mkdir(parents=True)
    write(args.output / "request.json", keys)
    write(
        args.output / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            args={k: str(v) for k, v in vars(args).items()},
            source_config_sha256=checked["config_sha256"],
            sample_manifest_sha256=checked["sample_manifest_sha256"],
            ids=[r["id"] for r in rows],
            task_counts=dict(Counter(r["question_type"] for r in rows)),
            label_counts=dict(Counter(r["target"] for r in rows)),
            unique_assets=len(keys),
            steps=512,
            seed=17,
            validation_inference=False,
            sealed_audit_loaded=False,
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
    records = []
    with (args.output / "worker.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            for i, key in enumerate(keys):
                record = read_tokens(child.stdout, (8, 16, 16, 768))
                if record is None or record[0] != str(i):
                    raise ValueError("worker order changed")
                bank[i] = record[1].reshape(2048, 768)
                delta = float(np.abs(bank[i].mean(0) - checked["feature_by_hash"][key]).max())
                if delta > 1e-5:
                    raise ValueError("pooled parity failed")
                records.append(
                    dict(
                        media_sha256=key,
                        pooled_max_abs=delta,
                        token_sha256=hashlib.sha256(bank[i].tobytes()).hexdigest(),
                    )
                )
            if read_tokens(child.stdout, (8, 16, 16, 768)) is not None or child.wait(30) != 0:
                raise ValueError("worker failed")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
    write(args.output / "extraction.json", records)
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
    text, labels = _text_features(rows, tokenizer, encoder, torch.device("cuda"))
    text, target = torch.stack(text).cuda(), torch.tensor(labels, device="cuda")
    del encoder, tokenizer
    video = torch.from_numpy(bank).cuda()
    indices = torch.tensor([keys.index(r["media_sha256"]) for r in rows], device="cuda")
    weights = _sample_weights(rows).cuda()
    initial_rng = torch.get_rng_state()
    results = {}

    def predict(model):
        model.eval()
        with torch.no_grad():
            return torch.cat(
                [model(video[indices[i : i + 8]], text[i : i + 8]).cpu() for i in range(0, 64, 8)]
            )

    for mode in ["attention", "no_video"]:
        out = args.output / mode
        out.mkdir()
        torch.set_rng_state(initial_rng)
        model = VideoTokenReadout(mode).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        history, norms = [], {}
        tick = time.perf_counter()
        for step in range(513):
            if step % 32 == 0:
                values = predict(model)
                entry = dict(
                    step=step,
                    metrics=_decision_metrics(list(values), labels, rows),
                    weighted_ce=float(
                        (
                            F.cross_entropy(values, target.cpu(), reduction="none") * weights.cpu()
                        ).sum()
                    ),
                    gradient_norms=norms,
                )
                if mode == "attention":
                    with torch.no_grad():
                        entropies = []
                        for i in range(0, 64, 8):
                            query = model.query(text[i : i + 8])
                            k = model.key(video[indices[i : i + 8]]) + model.position(
                                model.coordinates
                            )
                            prob = (torch.einsum("bch,bnh->bcn", query, k) / model.scale).softmax(
                                -1
                            )
                            entropies.append(
                                -(prob * prob.clamp_min(1e-30).log()).sum(-1).flatten()
                            )
                        entropy = torch.cat(entropies) / np.log(2048)
                        entry["normalized_attention_entropy"] = dict(
                            mean=float(entropy.mean()),
                            minimum=float(entropy.min()),
                            maximum=float(entropy.max()),
                        )
                history.append(entry)
                write(out / "history.json", history)
                print(mode, step, entry["metrics"]["macro_question_type"], flush=True)
            if step == 512:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, 64, 8):
                logits = model(video[indices[i : i + 8]], text[i : i + 8])
                loss = (
                    F.cross_entropy(logits, target[i : i + 8], reduction="none")
                    * weights[i : i + 8]
                ).sum()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
            norms = {
                n: float(p.grad.norm()) if p.grad is not None else None
                for n, p in model.named_parameters()
            }
            if any(v is not None and not np.isfinite(v) for v in norms.values()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
        torch.save(model.state_dict(), out / "final.pt")
        model.load_state_dict(torch.load(out / "final.pt", weights_only=True))
        if not torch.equal(values, predict(model)):
            raise ValueError("final reload mismatch")
        write(out / "predictions.json", _prediction_rows(rows, list(values), labels, mode))
        results[mode] = dict(
            final=history[-1],
            reload_exact=True,
            unique_examples=64,
            unique_assets=len(keys),
            encounters=512 * 64,
            repeated_encounters=511 * 64,
            checkpoint_sha256=sha(out / "final.pt"),
            seconds=time.perf_counter() - tick,
        )
    write(
        args.output / "result.json",
        dict(
            status="PASS_TRAIN_FIT_DIAGNOSTIC_ONLY",
            conditions=results,
            seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            cost_usd=0,
            limitation="Training subset only; no validation or generalization claim.",
        ),
    )


if __name__ == "__main__":
    main()
