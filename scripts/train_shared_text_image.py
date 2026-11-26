"""Same-image/readout comparison of independent versus shared text features."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_dinov2_image_probe import HASHES, TEXT, WEIGHTS, sha, write
from scripts.train_frozen_videoprism_clevr4_probe import (
    _option_embeddings,
    _prediction_rows,
    _quality_metrics,
    _taxonomy_metrics,
)
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.dinov2_features import (
    readout_features,
    require_image_splits,
)
from tiny_omni_decision.shared_text import shared_text_features

PARENT_HASHES = {
    "train-cells.npy": "34ad4961e4daf0df7e9672335b9d7c23a4a65d328491be9644e2631c8e376857",
    "val-cells.npy": "9fa89ef8474d34e4a5e12b6a65a911bbc8e1fc1f1006e568b42948de63e9e0bf",
    "config.json": "24ad7893c52f8a0f82c0065264211e6fb00a3e917189c05ac8fdf78f3426afe9",
    "global/history.json": "01f2a2798cd45d5d85020d88ca07550aed9236c701377ca70863dcce516f4cd0",
    "spatial/history.json": "1bc54cd23bb09a90d7b694e84925ea7b9e0ce93fb1de7b870e54cf4e1315c7a3",
}


def tensor_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


SHARED_BASE = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
SHARED_DELTA = "0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882"


def shared_options(device, model_path, checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    delta = torch.load(checkpoint, weights_only=True, map_location="cpu")
    expected = {
        n
        for n, _ in model.named_parameters()
        if n.startswith(("bert.encoder.layer.", "classifier."))
    }
    if set(delta) != expected:
        raise ValueError("shared checkpoint scope differs")
    model.load_state_dict(delta, strict=False)
    model.eval().requires_grad_(False).to(device)
    before = tensor_hash(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    keys = [f"{t}|{o}" for t, opts in CLEVR4_TAXONOMIES.items() for o in opts]
    prompts = [
        f"Image attribute: {t}. Candidate class: {o}."
        for t, opts in CLEVR4_TAXONOMIES.items()
        for o in opts
    ]
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(device)
    if encoded["input_ids"].shape[1] > 64:
        raise ValueError("shared prompts exceed fixed token budget")
    features = shared_text_features(model, encoded).cpu()
    if before != tensor_hash(model):
        raise ValueError("shared encoder mutated")
    return dict(zip(keys, features, strict=True)), model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "text-model", "cross-model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--epochs", type=int, choices=[24], default=24)
    p.add_argument("--feature-cache", type=Path, required=True)

    a = p.parse_args()
    shared_checkpoint = a.root / "numeric-deterministic-control-01/best-delta.pt"
    if (
        sha(a.cross_model / "model.safetensors") != SHARED_BASE
        or sha(shared_checkpoint) != SHARED_DELTA
    ):
        raise ValueError("shared text source changed")
    import shutil

    if shutil.disk_usage(a.root).free < 64 * 1024**2 + 512 * 1024**2:
        raise OSError("need 64MiB output allowance plus512MiB reserve")
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.epochs == 24 and a.feature_cache is None:
        raise ValueError("24-epoch experiment requires pinned parent cache")
    if a.feature_cache is not None:
        for name, expected_hash in PARENT_HASHES.items():
            if sha(a.feature_cache / name) != expected_hash:
                raise ValueError(f"parent changed: {name}")
    if (
        sha(a.model / "model.safetensors") != WEIGHTS
        or sha(a.text_model / "model.safetensors") != TEXT
    ):
        raise ValueError("model hash mismatch")
    directories = {
        "train": a.root / "clevr4-full-coverage-seed17",
        "val": a.root / "clevr4-sample-seed17-1024-512",
    }
    rows = {}
    manifests = {}
    for split, directory in directories.items():
        for kind in ["labels", "image-manifest"]:
            path = directory / f"{split}-{kind}.jsonl"
            if sha(path) != HASHES[path.name]:
                raise ValueError("data hash mismatch")
            data = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            (rows if kind == "labels" else manifests)[split] = data
        if [r["id"] for r in rows[split]] != [r["id"] for r in manifests[split]]:
            raise ValueError("ID order mismatch")
    require_image_splits(manifests["train"], manifests["val"])
    if a.smoke:
        rows = {k: v[:8] for k, v in rows.items()}
        manifests = {k: v[:8] for k, v in manifests.items()}
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    start = time.perf_counter()
    config = dict(
        argv={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        inputs=HASHES,
        seed=17,
        epochs=1 if a.smoke else a.epochs,
        parent_hashes=PARENT_HASHES if a.feature_cache is not None else None,
        normalization="cells /4, unchanged CLS; whole-image224x224",
        optimizer="AdamW lr1e-3 wd1e-4",
        batch_questions=64,
        selector="earliest min validation NLL",
        readout="bilinear",
        text_sources={"minilm": TEXT, "shared_base": SHARED_BASE, "shared_delta": SHARED_DELTA},
        torch=torch.__version__,
        model_files={x.name: sha(x) for x in a.model.iterdir() if x.is_file()},
    )
    write(a.output / "config.json", config)
    if a.feature_cache is not None:
        cells = {}
        parity = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for split in directories:
            values = np.load(a.feature_cache / f"{split}-cells.npy", allow_pickle=False)
            expected_count = 8424 if split == "train" else 512
            if values.shape != (expected_count, 17, 384) or not np.isfinite(values).all():
                raise ValueError("cache shape or values changed")
            cells[split] = values[: len(rows[split])]
            prior = np.load(
                a.root / "dinov2-clevr4-probe-01" / f"{split}-features.npy", allow_pickle=False
            )[: len(rows[split])]
            mean = np.concatenate([cells[split][:, 0], cells[split][:, 1:].mean(1)], 1)
            parity[split] = float(np.abs(mean - prior).max())
            if parity[split] > 1e-5:
                raise ValueError("global feature parity failed")
        extraction_seconds = time.perf_counter() - start
    targets = {
        s: [opts.index(str(r[t])) for r in data for t, opts in CLEVR4_TAXONOMIES.items()]
        for s, data in rows.items()
    }
    types = {s: list(CLEVR4_TAXONOMIES) * len(data) for s, data in rows.items()}
    results = {}
    initialization = None
    order_reference = None
    for condition in ["minilm", "shared"]:
        options, text_encoder = (
            _option_embeddings(device, a.text_model)
            if condition == "minilm"
            else shared_options(device, a.cross_model, shared_checkpoint)
        )
        text_parameters = sum(p.numel() for p in text_encoder.parameters())
        option_tensor = torch.stack(
            [
                torch.stack([options[f"{t}|{o}"] for o in opts])
                for t, opts in CLEVR4_TAXONOMIES.items()
            ]
        ).to(device)
        del text_encoder, options
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out = a.output / condition
        out.mkdir()
        np.save(out / "option-features.npy", option_tensor.cpu().numpy())
        features = {
            s: torch.from_numpy(readout_features(x, "spatial")).to(device) for s, x in cells.items()
        }
        torch.manual_seed(17)
        scorer = BilinearCandidateScorer(6528, 384, 128).to(device)
        trainable_parameters = sum(p.numel() for p in scorer.parameters())
        assert trainable_parameters == 884992
        initial = tensor_hash(scorer)
        if initialization is not None and initialization != initial:
            raise ValueError("initialization mismatch")
        initialization = initial
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.0001)
        rng = random.Random(17)
        order_hash = hashlib.sha256()
        history = []
        best = float("inf")
        trained_start = time.perf_counter()

        def logits_for(
            ids, split, rotate=False, _features=features, _scorer=scorer, _options=option_tensor
        ):
            ids = torch.tensor(ids, device=device)
            image_ids = ids // 4
            if rotate:
                image_ids = (image_ids + 1) % len(rows[split])
            visual = _features[split][image_ids, None, :].expand(-1, 10, -1)
            text = _options[ids % 4]
            return _scorer(torch.cat([visual, text], -1)).reshape(len(ids), 10)

        def predict(split, rotate=False, _scorer=scorer, _logits_for=logits_for):
            _scorer.eval()
            output = []
            with torch.inference_mode():
                for offset in range(0, len(targets[split]), 128):
                    ids = list(range(offset, min(offset + 128, len(targets[split]))))
                    output.extend(_logits_for(ids, split, rotate).cpu().unbind())
            return output

        for epoch in range(1, config["epochs"] + 1):
            order = list(range(len(targets["train"])))
            rng.shuffle(order)
            order_hash.update(np.asarray(order, dtype="<i8").tobytes())
            losses = []
            norms = []
            scorer.train()
            for offset in range(0, len(order), 64):
                ids = order[offset : offset + 64]
                logits = logits_for(ids, "train")
                label = torch.tensor([targets["train"][i] for i in ids], device=device)
                loss = F.cross_entropy(logits, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError("loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(p.grad) for p in scorer.parameters()])
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
                norms.append(float(norm))
            validation = predict("val")
            training = predict("train")
            quality = _quality_metrics(validation, targets["val"])
            record = dict(
                epoch=epoch,
                rolling_ce=float(np.mean(losses)),
                gradient_norm_mean=float(np.mean(norms)),
                train=_quality_metrics(training, targets["train"]),
                validation=quality,
                train_taxonomies=_taxonomy_metrics(training, targets["train"], types["train"]),
                val_taxonomies=_taxonomy_metrics(validation, targets["val"], types["val"]),
            )
            history.append(record)
            write(out / "history.json", history)
            if quality["nll"] < best:
                best = quality["nll"]
                best_epoch = epoch
                best_logits = [x.clone() for x in validation]
                torch.save(scorer.state_dict(), out / "best.pt")
            print(condition, epoch, quality["accuracy"], quality["nll"], flush=True)
        torch.save(scorer.state_dict(), out / "final.pt")
        scorer.load_state_dict(torch.load(out / "best.pt", weights_only=True, map_location=device))
        selected = predict("val")
        rotated = predict("val", True)
        assert all(torch.equal(x, y) for x, y in zip(selected, best_logits, strict=True))
        for name, values in [("matched", selected), ("rotated", rotated)]:
            write(
                out / f"{name}-predictions.json",
                _prediction_rows(rows["val"], values, targets["val"]),
            )
        order_digest = order_hash.hexdigest()
        if order_reference is not None and order_reference != order_digest:
            raise ValueError("sample order mismatch")
        order_reference = order_digest
        results[condition] = dict(
            best_epoch=best_epoch,
            text_parameters=text_parameters,
            image_pipeline_parameters=22056576 + text_parameters + trainable_parameters,
            option_features_sha256=sha(out / "option-features.npy"),
            matched=_quality_metrics(selected, targets["val"]),
            rotated=_quality_metrics(rotated, targets["val"]),
            matched_taxonomies=_taxonomy_metrics(selected, targets["val"], types["val"]),
            rotated_taxonomies=_taxonomy_metrics(rotated, targets["val"], types["val"]),
            initial_tensor_hash=initial,
            order_sha256=order_digest,
            best_sha256=sha(out / "best.pt"),
            final_sha256=sha(out / "final.pt"),
            seconds=time.perf_counter() - trained_start,
            reload_exact=True,
            unique_train_images=len(rows["train"]),
            unique_questions=len(targets["train"]),
            consumed=len(targets["train"]) * config["epochs"],
            selected_consumed=len(targets["train"]) * best_epoch,
        )
        write(out / "result.json", results[condition])
        del features, scorer, optimizer, predict, logits_for
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if (
        sha(a.model / "model.safetensors") != WEIGHTS
        or sha(a.text_model / "model.safetensors") != TEXT
    ):
        raise ValueError("frozen weights changed")
    if (
        sha(a.cross_model / "model.safetensors") != SHARED_BASE
        or sha(shared_checkpoint) != SHARED_DELTA
    ):
        raise ValueError("shared frozen source changed")
    write(
        a.output / "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            smoke=a.smoke,
            conditions=results,
            global_cache_parity=parity,
            extraction_seconds=extraction_seconds,
            total_seconds=time.perf_counter() - start,
            peak_vram_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            combined_runtime_implemented=False,
            combined_parameter_claim=False,
            trainable_parameters=trainable_parameters,
            frozen_weights_unchanged=True,
            device=str(device),
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
            cost_usd=0,
        ),
    )


if __name__ == "__main__":
    main()
