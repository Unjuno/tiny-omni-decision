"""Frozen question-conditioned ViLT image probe; full original options and split."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from scripts.audit_vilt_features import QUESTIONS
from scripts.train_frozen_text_probe import CandidateScorer
from scripts.train_frozen_videoprism_clevr4_probe import (
    _option_embeddings,
    _predict,
    _prediction_rows,
    _quality_metrics,
    _taxonomy_metrics,
)
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.vilt_compat import remove_legacy_position_ids

WEIGHTS = "4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19"
TEXT = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
HASHES = {
    "train-labels.jsonl": "00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e",
    "train-image-manifest.jsonl": (
        "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010"
    ),
    "val-labels.jsonl": "3c213f13ef2214baf1130cbe49c1d981b0125bbff3813e4ac5b4e341e443d4a1",
    "val-image-manifest.jsonl": "77faf433844be6b350cd0205fd3834cd8b028a6491ddd9eae0f6e9b7610cd38a",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n", encoding="utf-8")


def _tasks(rows, features_by_id, option_embeddings):
    tasks, targets, groups, taxonomies = [], [], [], []
    for row in rows:
        for index, (taxonomy, options) in enumerate(CLEVR4_TAXONOMIES.items()):
            target = str(row[taxonomy])
            if target not in options:
                raise ValueError("unknown target")
            image = features_by_id[row["id"]][index].expand(len(options), -1)
            candidate = torch.stack([option_embeddings[f"{taxonomy}|{o}"] for o in options])
            tasks.append(torch.cat([image, candidate], -1))
            targets.append(options.index(target))
            groups.append(row["id"])
            taxonomies.append(taxonomy)
    return tasks, targets, groups, taxonomies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "model", "text-model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--policy", choices=["bilinear"], required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if (
        sha(a.model / "pytorch_model.bin") != WEIGHTS
        or sha(a.text_model / "model.safetensors") != TEXT
    ):
        raise ValueError("weights mismatch")
    rows, manifests = {}, {}
    for split, directory, expected in [("train", a.train, 8424), ("val", a.validation, 512)]:
        for kind in ["labels", "image-manifest"]:
            path = directory / f"{split}-{kind}.jsonl"
            if sha(path) != HASHES[path.name]:
                raise ValueError(f"manifest mismatch {path}")
            data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(data) != expected:
                raise ValueError("wrong fixed corpus count")
            (rows if kind == "labels" else manifests)[split] = data
        if [r["id"] for r in rows[split]] != [r["id"] for r in manifests[split]]:
            raise ValueError("labels and media order mismatch")
    require_image_splits(manifests["train"], manifests["val"])
    if a.smoke:
        rows = {k: v[:8] for k, v in rows.items()}
        manifests = {k: v[:8] for k, v in manifests.items()}
    a.output.mkdir(parents=True)
    config = dict(
        argv={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha(Path(__file__)),
        started_utc=datetime.now(UTC).isoformat(),
        inputs=HASHES,
        seed=17,
        epochs=1 if a.smoke else 8,
        lr=0.001,
        weight_decay=0.0001,
        batch_questions=64,
        representation="CLS384 + mean-patch384",
        processor="pinned published ViLT whole-image resize",
        questions=QUESTIONS,
        selector="earliest minimum validation NLL",
        frozen_encoders=True,
        model_files={x.name: sha(x) for x in a.model.iterdir() if x.is_file()},
        versions={
            k: importlib.metadata.version(k) for k in ["torch", "transformers", "numpy", "Pillow"]
        },
    )
    write(a.output / "config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    random.seed(17)
    device = torch.device("cuda")
    start = time.perf_counter()
    features = {}
    if (
        shutil.disk_usage(a.output).free
        < (len(rows["train"]) + len(rows["val"])) * 4 * 768 * 4 + 1024**3
    ):
        raise ValueError("insufficient feature-cache disk reserve")
    model_config = ViltConfig.from_pretrained(a.model, local_files_only=True)
    model = ViltForQuestionAnswering(model_config)
    state = torch.load(a.model / "pytorch_model.bin", weights_only=True, map_location="cpu")
    model.load_state_dict(
        remove_legacy_position_ids(state, model_config.max_position_embeddings), strict=True
    )
    del state
    model.eval().requires_grad_(False).to(device)
    processor = ViltProcessor.from_pretrained(a.model, local_files_only=True)
    lengths = [len(x) for x in processor.tokenizer(list(QUESTIONS.values()))["input_ids"]]
    if max(lengths) > model_config.max_position_embeddings:
        raise ValueError("question exceeds position table")
    config["question_token_lengths"] = lengths
    config["cache_hashes"] = {}
    torch.cuda.reset_peak_memory_stats()
    for split, directory in [("train", a.train), ("val", a.validation)]:
        arrays = []
        for index, row in enumerate(manifests[split]):
            path = directory / split / "images" / (row["id"] + ".png")
            if sha(path) != row["sha256"]:
                raise ValueError("image changed")
            with Image.open(path) as source:
                image = source.convert("RGB")
            inputs = processor(
                images=[image] * 4,
                text=list(QUESTIONS.values()),
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                values = model.vilt(**inputs).pooler_output.cpu().numpy()
            if values.shape != (4, 768) or not np.isfinite(values).all():
                raise ValueError("invalid conditioned features")
            arrays.append(values)
            features[row["id"]] = torch.from_numpy(values.copy())
            if index % 256 == 0:
                print("features", split, index + 1, "/", len(manifests[split]), flush=True)
        cache = a.output / (split + "-features.npy")
        np.save(cache, np.stack(arrays))
        np.testing.assert_array_equal(np.load(cache, allow_pickle=False), np.stack(arrays))
        config["cache_hashes"][split] = sha(cache)
        write(a.output / "config.json", config)
    parameters = sum(p.numel() for p in model.parameters())
    extraction_seconds = time.perf_counter() - start
    extraction_peak = torch.cuda.max_memory_allocated()
    del model, processor, inputs, arrays, values
    torch.cuda.empty_cache()
    torch.manual_seed(17)
    option_embeddings, text_encoder = _option_embeddings(device, a.text_model)
    text_parameters = sum(p.numel() for p in text_encoder.parameters())
    train_tasks, train_targets, _, train_types = _tasks(rows["train"], features, option_embeddings)
    val_tasks, val_targets, _, val_types = _tasks(rows["val"], features, option_embeddings)
    rotated = dict(features)
    for i, row in enumerate(rows["val"]):
        rotated[row["id"]] = features[rows["val"][(i + 1) % len(rows["val"])]["id"]]
    rotated_tasks, _, _, _ = _tasks(rows["val"], rotated, option_embeddings)
    del text_encoder, option_embeddings, rotated, features
    torch.cuda.empty_cache()
    scorer = (
        CandidateScorer(1152, hidden_size=128)
        if a.policy == "concat"
        else BilinearCandidateScorer(768, 384, 128)
    ).to(device)
    readout_parameters = sum(p.numel() for p in scorer.parameters())
    assert readout_parameters == (147713 if a.policy == "concat" else 147712)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.0001)
    rng = random.Random(17)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    train_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config["epochs"] + 1):
        order = list(range(len(train_tasks)))
        rng.shuffle(order)
        scorer.train()
        losses = []
        for offset in range(0, len(order), 64):
            ids = order[offset : offset + 64]
            batch = torch.cat([train_tasks[i] for i in ids]).to(device)
            labels = torch.tensor([train_targets[i] for i in ids], device=device)
            scores = scorer(batch).reshape(len(ids), 10)
            loss = F.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in scorer.parameters()):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        logits = _predict(scorer, val_tasks, device)
        quality = _quality_metrics(logits, val_targets)
        train_logits = _predict(scorer, train_tasks, device)
        record = dict(
            epoch=epoch,
            train_ce=float(np.mean(losses)),
            train=_quality_metrics(train_logits, train_targets),
            validation=quality,
            per_taxonomy=_taxonomy_metrics(logits, val_targets, val_types),
        )
        history.append(record)
        if quality["nll"] < best_nll:
            best_nll = quality["nll"]
            best_epoch = epoch
            best_logits = [x.clone() for x in logits]
            torch.save(scorer.state_dict(), a.output / "best-readout.pt")
        write(a.output / "history.json", history)
        print(json.dumps(dict(epoch=epoch, validation=quality, best_epoch=best_epoch)), flush=True)
    torch.save(scorer.state_dict(), a.output / "final-readout.pt")
    scorer.load_state_dict(
        torch.load(a.output / "best-readout.pt", weights_only=True, map_location=device)
    )
    before = _predict(scorer, val_tasks, device)
    scorer.load_state_dict(
        torch.load(a.output / "best-readout.pt", weights_only=True, map_location=device)
    )
    after = _predict(scorer, val_tasks, device)
    assert all(torch.equal(x, y) for x, y in zip(before, after, strict=True))
    assert all(torch.equal(x, y) for x, y in zip(best_logits, after, strict=True))
    rotated_logits = _predict(scorer, rotated_tasks, device)
    for name, logits in [("matched", after), ("rotated", rotated_logits)]:
        write(
            a.output / f"{name}-predictions.json",
            _prediction_rows(rows["val"], logits, val_targets),
        )
    if (
        sha(a.model / "pytorch_model.bin") != WEIGHTS
        or sha(a.text_model / "model.safetensors") != TEXT
    ):
        raise ValueError("frozen input changed")
    result = dict(
        status="PASS_EXECUTION_ONLY",
        smoke=a.smoke,
        best_epoch=best_epoch,
        matched=_quality_metrics(after, val_targets),
        rotated=_quality_metrics(rotated_logits, val_targets),
        matched_taxonomies=_taxonomy_metrics(after, val_targets, val_types),
        rotated_taxonomies=_taxonomy_metrics(rotated_logits, val_targets, val_types),
        encoder_parameters=parameters,
        text_parameters=text_parameters,
        readout_parameters=readout_parameters,
        total_parameters=parameters + text_parameters + readout_parameters,
        unique_train_images=len(rows["train"]),
        unique_train_questions=len(train_tasks),
        consumed_questions=len(train_tasks) * config["epochs"],
        selected_consumed_questions=len(train_tasks) * best_epoch,
        extraction_seconds=extraction_seconds,
        training_evaluation_seconds=time.perf_counter() - train_start,
        total_seconds=time.perf_counter() - start,
        peak_vram_bytes=max(extraction_peak, torch.cuda.max_memory_allocated()),
        best_sha256=sha(a.output / "best-readout.pt"),
        final_sha256=sha(a.output / "final-readout.pt"),
        frozen_weights_unchanged=True,
        reload_exact=True,
        gpu=torch.cuda.get_device_name(),
        cost_usd=0,
        finished_utc=datetime.now(UTC).isoformat(),
        exact_resume=False,
    )
    write(a.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
