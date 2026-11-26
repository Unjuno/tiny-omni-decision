"""Extract pinned V-JEPA image features and train a small CLEVR-4 readout."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-sha256", required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-images", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_vjepa(args: argparse.Namespace, torch_module: Any):
    expected_commit = "204698b45b3712590f06245fbfba32d3be539812"
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.vjepa_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError(f"unexpected V-JEPA source commit: {commit}")
    if _sha256(args.vjepa_checkpoint) != args.vjepa_sha256:
        raise ValueError("V-JEPA checkpoint hash did not match the pinned SHA-256")
    sys.path.insert(0, str(args.vjepa_source.resolve()))
    from app.vjepa_2_1.models import vision_transformer
    from evals.hub.preprocessor import vjepa2_preprocessor

    checkpoint = torch_module.load(
        args.vjepa_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    state = checkpoint["ema_encoder"]
    clean_state = {
        key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()
    }
    encoder = vision_transformer.vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        n_output_distillation=1,
    )
    encoder.load_state_dict(clean_state, strict=True)
    del checkpoint, state, clean_state
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, vjepa2_preprocessor(pretrained=False, crop_size=384), commit


def _extract_image_features(
    rows: list[dict[str, Any]],
    split: str,
    root: Path,
    encoder: nn.Module,
    transform: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], Tensor, list[str], float]:
    ids = [str(row["id"]) for row in rows]
    asset_hashes = []
    tensors = []
    started = time.perf_counter()
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        images = []
        for image_id in batch_ids:
            path = root / split / "images" / f"{image_id}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            asset_hashes.append(_sha256(path))
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(transform([np.asarray(rgb)])[0])
        batch = torch.stack(images).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            tokens = encoder(batch)
            pooled = tokens.float().mean(dim=1)
        if pooled.shape[-1] != 768 or not torch.isfinite(pooled).all():
            raise ValueError("V-JEPA produced invalid image features")
        tensors.append(pooled.cpu())
    return ids, torch.cat(tensors), asset_hashes, time.perf_counter() - started


def _option_embeddings(
    tokenizer: Any, model: nn.Module, device: torch.device
) -> tuple[dict[str, Tensor], float]:
    strings = {
        f"{taxonomy}|{option}": f"Image attribute: {taxonomy}. Candidate class: {option}."
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    }
    keys = list(strings)
    encoded = tokenizer(
        [strings[key] for key in keys],
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu()
    if not torch.isfinite(pooled).all():
        raise ValueError("text candidate embeddings contain NaN or Inf")
    return dict(zip(keys, pooled, strict=True)), int(model.config.hidden_size)


def _tasks(
    rows: list[dict[str, Any]], image_features: Tensor, option_embeddings: dict[str, Tensor]
) -> tuple[list[Tensor], list[int], list[str], list[str]]:
    features = []
    targets = []
    groups = []
    taxonomies = []
    for index, row in enumerate(rows):
        image_id = str(row["id"])
        for taxonomy, options in CLEVR4_TAXONOMIES.items():
            target = str(row[taxonomy])
            if target not in options:
                raise ValueError(f"unknown CLEVR-4 {taxonomy} label {target!r}")
            targets.append(options.index(target))
            groups.append(image_id)
            taxonomies.append(taxonomy)
            image_rows = image_features[index].expand(len(options), -1)
            text_rows = torch.stack(
                [option_embeddings[f"{taxonomy}|{option}"] for option in options]
            )
            features.append(torch.cat([image_rows, text_rows], dim=-1))
    return features, targets, groups, taxonomies


def _predict(scorer: nn.Module, features: list[Tensor], device: torch.device) -> list[Tensor]:
    scorer.eval()
    with torch.inference_mode():
        widths = [item.shape[0] for item in features]
        logits = scorer(torch.cat(features).to(device)).cpu()
    outputs = []
    offset = 0
    for width in widths:
        outputs.append(logits[offset : offset + width])
        offset += width
    return outputs


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this pinned V-JEPA feature run requires the local CUDA device")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)

    sample_root = args.sample_dir.resolve()
    train_rows = _load_rows(sample_root / "train-labels.jsonl")
    validation_rows = _load_rows(sample_root / "val-labels.jsonl")
    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("CLEVR-4 image IDs overlap between train and validation")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("train labels include a non-train source row")
    if any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("validation labels include a non-validation source row")
    train_assets = _load_rows(sample_root / "train-image-manifest.jsonl")
    validation_assets = _load_rows(sample_root / "val-image-manifest.jsonl")
    train_asset_hashes = {row["sha256"] for row in train_assets}
    validation_asset_hashes = {row["sha256"] for row in validation_assets}
    if train_asset_hashes & validation_asset_hashes:
        raise ValueError("CLEVR-4 media content hashes overlap between splits")

    encoder, transform, vjepa_commit = _load_vjepa(args, torch)
    encoder = encoder.to(device)
    train_feature_ids, train_image_features, train_image_hashes, train_seconds = (
        _extract_image_features(
            train_rows, "train", sample_root, encoder, transform, device, args.batch_images
        )
    )
    val_feature_ids, val_image_features, val_image_hashes, val_seconds = _extract_image_features(
        validation_rows, "val", sample_root, encoder, transform, device, args.batch_images
    )
    image_feature_path = output / "image-features.pt"
    torch.save(
        {
            "source_revision": "sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73",
            "encoder_commit": vjepa_commit,
            "encoder_sha256": args.vjepa_sha256,
            "train_ids": train_feature_ids,
            "validation_ids": val_feature_ids,
            "train_features": train_image_features,
            "validation_features": val_image_features,
            "train_asset_sha256": train_image_hashes,
            "validation_asset_sha256": val_image_hashes,
        },
        image_feature_path,
    )
    del encoder
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=True).eval().to(device)
    for parameter in text_model.parameters():
        parameter.requires_grad_(False)
    option_embeddings, text_hidden = _option_embeddings(tokenizer, text_model, device)
    train_features, train_targets, train_groups, train_taxonomies = _tasks(
        train_rows, train_image_features, option_embeddings
    )
    val_features, val_targets, val_groups, val_taxonomies = _tasks(
        validation_rows, val_image_features, option_embeddings
    )
    del text_model
    torch.cuda.empty_cache()

    scorer = CandidateScorer(768 + text_hidden).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    best_nll = float("inf")
    best_state = None
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_features)))
        rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            batch_ids = order[start : start + args.batch_questions]
            rows = [train_features[index].to(device) for index in batch_ids]
            targets = torch.tensor([train_targets[index] for index in batch_ids], device=device)
            widths = [row.shape[0] for row in rows]
            scores = scorer(torch.cat(rows))
            padded = scores.new_full((len(rows), max(widths)), -1e4)
            offset = 0
            for row_index, width in enumerate(widths):
                padded[row_index, :width] = scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(padded, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _predict(scorer, val_features, device)
        aggregate = _metrics(val_logits, val_targets)
        per_taxonomy = {
            taxonomy: _metrics(
                [
                    logit
                    for logit, group in zip(val_logits, val_taxonomies, strict=True)
                    if group == taxonomy
                ],
                [
                    target
                    for target, group in zip(val_targets, val_taxonomies, strict=True)
                    if group == taxonomy
                ],
            )
            for taxonomy in CLEVR4_TAXONOMIES
        }
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                "validation": aggregate,
                "validation_by_taxonomy": per_taxonomy,
            }
        )
        if aggregate["nll"] < best_nll:
            best_nll = aggregate["nll"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("no validation-selected image readout checkpoint was produced")
    scorer.load_state_dict(best_state)
    val_logits = _predict(scorer, val_features, device)
    selected = _metrics(val_logits, val_targets)
    per_taxonomy = {
        taxonomy: _metrics(
            [
                logit
                for logit, group in zip(val_logits, val_taxonomies, strict=True)
                if group == taxonomy
            ],
            [
                target
                for target, group in zip(val_targets, val_taxonomies, strict=True)
                if group == taxonomy
            ],
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }
    torch.save(best_state, output / "best-readout.pt")
    predictions = []
    for row_index, row in enumerate(validation_rows):
        for taxonomy_index, taxonomy in enumerate(CLEVR4_TAXONOMIES):
            task_index = row_index * len(CLEVR4_TAXONOMIES) + taxonomy_index
            options = CLEVR4_TAXONOMIES[taxonomy]
            probs = torch.softmax(val_logits[task_index].float(), dim=-1)
            predictions.append(
                {
                    "id": f"{row['id']}:{taxonomy}",
                    "image_id": row["id"],
                    "taxonomy": taxonomy,
                    "target": options[val_targets[task_index]],
                    "prediction": options[int(probs.argmax())],
                    "options": options,
                    "probabilities": probs.tolist(),
                }
            )
    (output / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "status": "complete_image_only_synthetic_probe",
        "task_claim": (
            "CLEVR-4 is synthetic image taxonomy classification, not natural-image understanding."
        ),
        "seed": args.seed,
        "source_revision": "sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73",
        "source_archive_sha512_expected": (
            "769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057"
            "ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a"
        ),
        "source_archive_full_hash_verified": False,
        "training_image_count": len(train_rows),
        "validation_image_count": len(validation_rows),
        "training_unique_image_ids": len(set(train_feature_ids)),
        "validation_unique_image_ids": len(set(val_feature_ids)),
        "train_validation_image_id_overlap": 0,
        "train_validation_media_hash_overlap": 0,
        "training_task_count": len(train_features),
        "validation_task_count": len(val_features),
        "task_types": list(CLEVR4_TAXONOMIES),
        "train_image_feature_hashes": train_image_hashes,
        "validation_image_feature_hashes": val_image_hashes,
        "image_feature_cache_bytes": image_feature_path.stat().st_size,
        "image_feature_cache_sha256": _sha256(image_feature_path),
        "vjepa_source_revision": vjepa_commit,
        "vjepa_checkpoint_sha256": args.vjepa_sha256,
        "text_model_revision": args.text_revision,
        "text_model_weights_sha256": _sha256(args.text_model / "model.safetensors"),
        "readout_parameters": trainable_parameters,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "epochs": args.epochs,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["validation"]["nll"])["epoch"],
        "validation_selected_metrics": selected,
        "validation_by_taxonomy": per_taxonomy,
        "train_feature_extraction_seconds": train_seconds,
        "validation_feature_extraction_seconds": val_seconds,
        "training_seconds": training_seconds,
        "device": str(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "note": "Checkpoint selection uses the named validation split; no sealed audit is loaded.",
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
